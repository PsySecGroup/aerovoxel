/**
 * @file ray_voxel.cpp
 * @brief Multi-camera motion-based voxel reconstruction (optimized build).
 *
 * CHANGES FROM v1
 * ---------------
 * 1. image.scale in profile.json
 *    Downsamples frames before motion detection. 0.25 = quarter-res = 16x fewer pixels.
 *    Recommended for calibration. Add to profile.json:
 *      "image": { "scale": 0.25 },
 *      "grid":  { "resolution": 100 }
 *
 * 2. Dense std::vector<float> voxel grid
 *    Replaces std::unordered_map. No hash overhead; contiguous memory = cache-friendly.
 *    Memory: N^3 * 4 bytes.  N=100 = 4 MB, N=200 = 32 MB, N=500 = 500 MB.
 *    For calibration, resolution=100 is strongly recommended.
 *
 * 3. cast_ray_and_accumulate()
 *    Inline DDA with direct atomic grid writes. Eliminates one std::vector<RayStep>
 *    heap alloc/dealloc per changed pixel. At 1080p / 10% motion ≈ 200K saved per pair.
 *
 * 4. OpenMP throughout
 *    Pixel loop, detect_motion(), and downscale_image() all parallelized.
 *    Build: g++ -O3 -fopenmp -I vendor -o ray_voxel ray_voxel.cpp
 *
 * 5. Timing instrumentation
 *    Per-stage breakdowns printed to stdout. Useful for profiling on target hardware.
 *
 * 6. thread_local RNG in load_image_gray
 *    Safe for parallel image loads (original used shared static mt19937).
 *
 * Output format: unchanged — backward-compatible sparse binary.
 *
 * Usage: ray_voxel <metadata.json> <image_folder> <output.bin>
 */

#include <iostream>
#include <fstream>
#include <cmath>
#include <limits>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <chrono>
#include <climits>
#include <random>
#include <omp.h>

#include "nlohmann/json.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using json = nlohmann::json;


// ============================================================
// Timing helper
// ============================================================

using HRC = std::chrono::high_resolution_clock;

static inline double ms_since(const HRC::time_point& t0) {
    return std::chrono::duration<double, std::milli>(HRC::now() - t0).count();
}


// ============================================================
// Section 1: Core Data Structures
// ============================================================

struct Vec3 { float x, y, z; };
struct Mat3 { float m[9]; };

struct FrameInfo {
    int   camera_index    = 0;
    int   frame_index     = 0;
    Vec3  camera_position = {0, 0, 0};
    float yaw = 0, pitch = 0, roll = 0, fov_degrees = 60;
    std::string image_file;
};


// ============================================================
// Section 2: Math Helpers
// ============================================================

static inline float deg2rad(float d) { return d * 3.14159265358979323846f / 180.f; }

static inline Vec3 normalize(const Vec3& v) {
    float len = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    return (len < 1e-12f) ? Vec3{0, 0, 0} : Vec3{v.x/len, v.y/len, v.z/len};
}

static inline Vec3 mat3_mul_vec3(const Mat3& M, const Vec3& v) {
    return { M.m[0]*v.x + M.m[1]*v.y + M.m[2]*v.z,
             M.m[3]*v.x + M.m[4]*v.y + M.m[5]*v.z,
             M.m[6]*v.x + M.m[7]*v.y + M.m[8]*v.z };
}


// ============================================================
// Section 3: Euler Angles → Rotation Matrix (original convention)
// ============================================================

Mat3 rotation_matrix_yaw_pitch_roll(float yaw_deg, float pitch_deg, float roll_deg) {
    float y = deg2rad(yaw_deg), p = deg2rad(pitch_deg), r = deg2rad(roll_deg);
    float cy=std::cos(y), sy=std::sin(y);
    float cp=std::cos(p), sp=std::sin(p);
    float cr=std::cos(r), sr=std::sin(r);
    float Rz[9]={ cy,-sy,0, sy,cy,0, 0,0,1 };
    float Ry[9]={ cr,0,sr,  0,1,0,  -sr,0,cr };
    float Rx[9]={ 1,0,0,    0,cp,-sp, 0,sp,cp };
    auto mm3 = [](const float A[9], const float B[9], float C[9]) {
        for(int row=0;row<3;row++) for(int col=0;col<3;col++)
            C[row*3+col] = A[row*3+0]*B[0*3+col]
                         + A[row*3+1]*B[1*3+col]
                         + A[row*3+2]*B[2*3+col];
    };
    float Rt[9], Rf[9];
    mm3(Rz, Ry, Rt);
    mm3(Rt, Rx, Rf);
    Mat3 out; for(int i=0;i<9;i++) out.m[i]=Rf[i]; return out;
}


// ============================================================
// Section 3b: Generator-Convention Camera Matrix
// ============================================================

/**
 * @brief Build a camera-to-world rotation matrix matching generator.py's convention.
 * Y-forward, Z-up. Yaw=0 looks in +Y, yaw=90 looks in +X.
 */
Mat3 camera_matrix_from_generator_convention(float yaw_deg, float pitch_deg, float roll_deg) {
    float y = deg2rad(yaw_deg), p = deg2rad(pitch_deg);
    Vec3 F = normalize({ std::cos(p)*std::sin(y), std::cos(p)*std::cos(y), std::sin(p) });
    Vec3 world_up = {0, 0, 1};
    Vec3 right = normalize({
        F.y*world_up.z - F.z*world_up.y,
        F.z*world_up.x - F.x*world_up.z,
        F.x*world_up.y - F.y*world_up.x
    });
    float dot = F.x*world_up.x + F.y*world_up.y + F.z*world_up.z;
    if(std::fabs(dot) > 0.999f) right = normalize({1, 0, 0});
    Vec3 local_up = normalize({
        right.y*F.z - right.z*F.y,
        right.z*F.x - right.x*F.z,
        right.x*F.y - right.y*F.x
    });
    if(std::fabs(roll_deg) > 1e-4f) {
        float r = deg2rad(roll_deg), cr=std::cos(r), sr=std::sin(r);
        Vec3 nr={ cr*right.x+sr*local_up.x, cr*right.y+sr*local_up.y, cr*right.z+sr*local_up.z };
        Vec3 nu={-sr*right.x+cr*local_up.x,-sr*right.y+cr*local_up.y,-sr*right.z+cr*local_up.z };
        right = nr; local_up = nu;
    }
    Mat3 M;
    M.m[0]=right.x;   M.m[1]=local_up.x;  M.m[2]=-F.x;
    M.m[3]=right.y;   M.m[4]=local_up.y;  M.m[5]=-F.y;
    M.m[6]=right.z;   M.m[7]=local_up.z;  M.m[8]=-F.z;
    return M;
}


// ============================================================
// Section 3c: Profile  (MODIFIED: added image_scale)
// ============================================================

/**
 * @brief All tunable parameters for a single scene run.
 *
 * profile.json schema additions in this build:
 * @code
 * {
 *   "image": {
 *     "scale": 0.25   // linear downsample factor before motion detection.
 *                     // 0.25 = quarter-res = 16x fewer pixels to cast rays for.
 *                     // Recommended for calibration.  Default 1.0 = no scaling.
 *   },
 *   "grid": {
 *     "resolution": 100  // Use 100 for calibration (4 MB), 500 for production (500 MB).
 *   }
 * }
 * @endcode
 */
struct Profile {
    // grid
    int   resolution   = 500;
    float voxel_size   = 6.f;
    bool  center_auto  = true;
    Vec3  grid_center  = {0, 0, 500};
    // motion
    float pixel_diff_threshold  = 2.f;
    // ray
    int   max_steps             = 0;
    int   frame_stride          = 1;
    float min_start_distance    = -1.f;
    float near_field_suppression = 0.f;
    float max_useful_distance   = -1.f;
    // image (NEW)
    float image_scale = 1.f;   ///< Downsample factor applied before motion detection.
                                ///< 0.25 = quarter linear resolution = 16x fewer pixels.
                                ///< Recommended calibration setting: 0.25.
    // per-camera overrides
    struct CameraOverride {
        float pixel_diff_threshold = -1.f;
        int   frame_stride         = -1;
    };
    std::map<int, CameraOverride> camera_overrides;
};

Profile load_profile(const std::string& metadata_path) {
    Profile prof;
    std::string dir = metadata_path;
    auto slash = dir.find_last_of("/\\");
    dir = (slash != std::string::npos) ? dir.substr(0, slash) : ".";
    std::string path = dir + "/profile.json";

    std::ifstream ifs(path);
    if(!ifs.is_open()) {
        std::cout << "[Profile] No profile.json at " << path << " — using defaults.\n";
        return prof;
    }
    json j;
    try { ifs >> j; } catch(const std::exception& e) {
        std::cerr << "[Profile] Parse error: " << e.what() << " — using defaults.\n";
        return prof;
    }
    std::cout << "[Profile] Loaded " << path << "\n";

    if(j.contains("grid")) {
        auto& g = j["grid"];
        if(g.contains("resolution")) prof.resolution = g["resolution"].get<int>();
        if(g.contains("voxel_size")) prof.voxel_size  = g["voxel_size"].get<float>();
        if(g.contains("center")) {
            auto& c = g["center"];
            if(c.is_string() && c.get<std::string>() == "auto") {
                prof.center_auto = true;
            } else if(c.is_array() && c.size() == 3) {
                prof.center_auto   = false;
                prof.grid_center.x = c[0].get<float>();
                prof.grid_center.y = c[1].get<float>();
                prof.grid_center.z = c[2].get<float>();
            }
        }
    }
    if(j.contains("motion")) {
        auto& m = j["motion"];
        if(m.contains("pixel_diff_threshold"))
            prof.pixel_diff_threshold = m["pixel_diff_threshold"].get<float>();
    }
    if(j.contains("ray")) {
        auto& r = j["ray"];
        if(r.contains("max_steps"))              prof.max_steps              = r["max_steps"].get<int>();
        if(r.contains("frame_stride"))            prof.frame_stride           = r["frame_stride"].get<int>();
        if(r.contains("min_start_distance"))      prof.min_start_distance     = r["min_start_distance"].get<float>();
        if(r.contains("near_field_suppression"))  prof.near_field_suppression = r["near_field_suppression"].get<float>();
        if(r.contains("max_useful_distance"))     prof.max_useful_distance    = r["max_useful_distance"].get<float>();
    }
    // NEW: image section
    if(j.contains("image")) {
        auto& img = j["image"];
        if(img.contains("scale"))
            prof.image_scale = std::max(0.01f, std::min(1.0f, img["scale"].get<float>()));
    }
    if(j.contains("cameras") && j["cameras"].is_object()) {
        for(auto& [key, val] : j["cameras"].items()) {
            int cam_id = std::stoi(key);
            Profile::CameraOverride ov;
            if(val.contains("pixel_diff_threshold"))
                ov.pixel_diff_threshold = val["pixel_diff_threshold"].get<float>();
            if(val.contains("frame_stride"))
                ov.frame_stride = val["frame_stride"].get<int>();
            prof.camera_overrides[cam_id] = ov;
        }
    }
    return prof;
}


// ============================================================
// Section 4: JSON Metadata Loading — unchanged
// ============================================================

std::vector<FrameInfo> load_metadata(const std::string& json_path) {
    std::vector<FrameInfo> frames;
    std::ifstream ifs(json_path);
    if(!ifs.is_open()) { std::cerr << "ERROR: Cannot open " << json_path << "\n"; return frames; }
    json j; ifs >> j;
    if(!j.is_array()) { std::cerr << "ERROR: JSON top level is not an array.\n"; return frames; }
    for(const auto& entry : j) {
        FrameInfo fi;
        fi.camera_index = entry.value("camera_index", 0);
        fi.frame_index  = entry.value("frame_index",  0);
        fi.yaw          = entry.value("yaw",          0.f);
        fi.pitch        = entry.value("pitch",        0.f);
        fi.roll         = entry.value("roll",         0.f);
        fi.fov_degrees  = entry.value("fov_degrees",  60.f);
        fi.image_file   = entry.value("image_file",   "");
        if(entry.contains("camera_position") && entry["camera_position"].is_array()) {
            auto arr = entry["camera_position"];
            if(arr.size() >= 3)
                fi.camera_position = { arr[0].get<float>(), arr[1].get<float>(), arr[2].get<float>() };
        }
        frames.push_back(fi);
    }
    return frames;
}


// ============================================================
// Section 5: Image Loading, Downscaling, Motion Detection
// ============================================================

struct ImageGray {
    int width = 0, height = 0;
    std::vector<float> pixels;
};

/**
 * @brief Load grayscale float image with dithering noise.
 *
 * Uses thread_local RNG rather than the original shared static mt19937,
 * making this safe to call from multiple OMP threads simultaneously if needed.
 */
bool load_image_gray(const std::string& img_path, ImageGray& out) {
    int w, h, ch;
    unsigned char* data = stbi_load(img_path.c_str(), &w, &h, &ch, 1);
    if(!data) { std::cerr << "Failed to load image: " << img_path << "\n"; return false; }
    out.width  = w;
    out.height = h;
    out.pixels.resize(w * h);
    // thread_local: each thread gets its own RNG state, no shared-state race.
    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> noise(-1.f, 1.f);
    for(int i = 0; i < w*h; i++) {
        float v = static_cast<float>(data[i]) + noise(gen);
        out.pixels[i] = std::max(0.f, std::min(255.f, v));
    }
    stbi_image_free(data);
    return true;
}

/**
 * @brief Fast box-filter downsample.
 *
 * Reduces to (width*scale) x (height*scale) by averaging each output pixel's
 * source region. Parallelized via OpenMP — at 24 MP this runs in a few ms.
 *
 * The FOV angle is preserved: focal_len is recalculated from mm.width after
 * downscaling, maintaining correct ray directions at the reduced resolution.
 *
 * @param src   Source image.
 * @param scale Linear scale factor in (0, 1]. Returns src copy if >= 1.
 */
static ImageGray downscale_image(const ImageGray& src, float scale) {
    if(scale >= 1.f - 1e-4f || src.width == 0) return src;
    int dw = std::max(1, (int)std::round(src.width  * scale));
    int dh = std::max(1, (int)std::round(src.height * scale));
    float sx = (float)src.width  / (float)dw;
    float sy = (float)src.height / (float)dh;
    ImageGray out;
    out.width  = dw;
    out.height = dh;
    out.pixels.assign(dw * dh, 0.f);
    #pragma omp parallel for schedule(static)
    for(int oy = 0; oy < dh; oy++) {
        int y0 = (int)(oy * sy);
        int y1 = std::min((int)std::ceil((oy+1)*sy) - 1, src.height - 1);
        for(int ox = 0; ox < dw; ox++) {
            int   x0  = (int)(ox * sx);
            int   x1  = std::min((int)std::ceil((ox+1)*sx) - 1, src.width - 1);
            float sum = 0.f; int cnt = 0;
            for(int y = y0; y <= y1; y++)
                for(int x = x0; x <= x1; x++)
                    { sum += src.pixels[y * src.width + x]; cnt++; }
            out.pixels[oy * dw + ox] = cnt > 0 ? sum / cnt : 0.f;
        }
    }
    return out;
}

/**
 * @brief Per-pixel motion detection result.
 *
 * changed is std::vector<uint8_t> rather than vector<bool>.
 * vector<bool> uses packed bit storage: concurrent writes to different indices
 * from separate OMP threads can corrupt adjacent bits (false sharing on bytes).
 * uint8_t stores one byte per element — no sharing, safe for parallel writes.
 */
struct MotionMask {
    int width = 0, height = 0;
    std::vector<uint8_t> changed;
    std::vector<float>   diff;
};

/**
 * @brief Compute per-pixel motion mask. Parallelized via OpenMP.
 */
MotionMask detect_motion(const ImageGray& prev, const ImageGray& next, float threshold) {
    MotionMask mm;
    if(prev.width != next.width || prev.height != next.height) {
        std::cerr << "Size mismatch in detect_motion\n"; return mm;
    }
    mm.width   = prev.width;
    mm.height  = prev.height;
    const int total = mm.width * mm.height;
    mm.changed.assign(total, 0);
    mm.diff.assign(total, 0.f);
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < total; i++) {
        float d       = std::fabs(prev.pixels[i] - next.pixels[i]);
        mm.diff[i]    = d;
        mm.changed[i] = (d > threshold) ? 1u : 0u;
    }
    return mm;
}


// ============================================================
// Section 6: DDA Voxel Ray Casting
// ============================================================

struct RayStep {
    int ix, iy, iz, step_count;
    float distance;
};

static inline float safe_div(float num, float den) {
    return (std::fabs(den) < 1e-12f) ? std::numeric_limits<float>::infinity() : num/den;
}

/**
 * @brief Original ray traversal returning a list of visited voxels.
 *
 * Retained for debugging and external visualization tools.
 * NOT used in the main processing pipeline — see cast_ray_and_accumulate().
 */
std::vector<RayStep> cast_ray_into_grid(
    const Vec3& camera_pos, const Vec3& dir,
    int N, float voxel_size, const Vec3& grid_center)
{
    std::vector<RayStep> steps;
    steps.reserve(64);
    float hs = 0.5f * N * voxel_size;
    Vec3 gmin = { grid_center.x-hs, grid_center.y-hs, grid_center.z-hs };
    Vec3 gmax = { grid_center.x+hs, grid_center.y+hs, grid_center.z+hs };
    float t_min=0, t_max=std::numeric_limits<float>::infinity();
    for(int i=0;i<3;i++){
        float o=(i==0)?camera_pos.x:((i==1)?camera_pos.y:camera_pos.z);
        float d=(i==0)?dir.x:((i==1)?dir.y:dir.z);
        float mn=(i==0)?gmin.x:((i==1)?gmin.y:gmin.z);
        float mx=(i==0)?gmax.x:((i==1)?gmax.y:gmax.z);
        if(std::fabs(d)<1e-12f){ if(o<mn||o>mx) return steps; }
        else{
            float t1=(mn-o)/d, t2=(mx-o)/d;
            float tn=std::fmin(t1,t2), tf=std::fmax(t1,t2);
            if(tn>t_min) t_min=tn; if(tf<t_max) t_max=tf;
            if(t_min>t_max) return steps;
        }
    }
    if(t_min<0) t_min=0;
    Vec3 sw={camera_pos.x+t_min*dir.x, camera_pos.y+t_min*dir.y, camera_pos.z+t_min*dir.z};
    int ix=(int)((sw.x-gmin.x)/voxel_size);
    int iy=(int)((sw.y-gmin.y)/voxel_size);
    int iz=(int)((sw.z-gmin.z)/voxel_size);
    if(ix<0||ix>=N||iy<0||iy>=N||iz<0||iz>=N) return steps;
    int sx=(dir.x>=0)?1:-1, sy=(dir.y>=0)?1:-1, sz=(dir.z>=0)?1:-1;
    int nx=ix+(sx>0?1:0), ny=iy+(sy>0?1:0), nz=iz+(sz>0?1:0);
    float tmx=safe_div(gmin.x+nx*voxel_size-camera_pos.x,dir.x);
    float tmy=safe_div(gmin.y+ny*voxel_size-camera_pos.y,dir.y);
    float tmz=safe_div(gmin.z+nz*voxel_size-camera_pos.z,dir.z);
    float tdx=safe_div(voxel_size,std::fabs(dir.x));
    float tdy=safe_div(voxel_size,std::fabs(dir.y));
    float tdz=safe_div(voxel_size,std::fabs(dir.z));
    float tc=t_min; int sc=0;
    while(tc<=t_max){
        steps.push_back({ix,iy,iz,sc,tc});
        if(tmx<tmy&&tmx<tmz){ix+=sx;tc=tmx;tmx+=tdx;}
        else if(tmy<tmz)     {iy+=sy;tc=tmy;tmy+=tdy;}
        else                  {iz+=sz;tc=tmz;tmz+=tdz;}
        sc++;
        if(ix<0||ix>=N||iy<0||iy>=N||iz<0||iz>=N) break;
    }
    return steps;
}

/**
 * @brief Inline DDA traversal with direct atomic voxel accumulation.
 *
 * This replaces the cast_ray_into_grid() + accumulation loop pattern used in
 * the original hot path.  The key difference: no heap allocation at all.
 * The original code allocated one std::vector<RayStep> per changed pixel —
 * at 1080p / 10% motion that's ~200K alloc/dealloc cycles per frame pair,
 * each touching the global allocator.  This function eliminates that entirely.
 *
 * Thread-safety: the only shared mutable state is voxel_grid[], accessed via
 * #pragma omp atomic update.  All DDA state variables are local to this call.
 * Contention is low for normal calibration scenes where rays fan out across
 * diverse voxel indices.
 *
 * Behaviour is numerically identical to the original cast_ray_into_grid +
 * accumulation loop — same AABB slab intersection, same DDA step sequence,
 * same near-field suppression ramp, same distance check.
 *
 * @param camera_pos      Ray origin in world space.
 * @param dir_normalized  Unit-length world-space ray direction.
 * @param N               Grid resolution (voxels per axis).
 * @param voxel_size      Voxel side length in world units.
 * @param grid_center     World-space centre of the voxel volume.
 * @param pix_val         Motion signal strength (raw pixel difference).
 * @param prof            Resolved profile parameters (read-only).
 * @param voxel_grid      Dense flat grid indexed ix*N*N + iy*N + iz (shared, atomic).
 */
static void cast_ray_and_accumulate(
    const Vec3& camera_pos,
    const Vec3& dir_normalized,
    int N, float voxel_size, const Vec3& grid_center,
    float pix_val, const Profile& prof,
    std::vector<float>& voxel_grid)
{
    // Phase 1: AABB slab intersection (same logic as cast_ray_into_grid).
    const float hs = 0.5f * N * voxel_size;
    const Vec3 gmin = { grid_center.x-hs, grid_center.y-hs, grid_center.z-hs };
    const Vec3 gmax = { grid_center.x+hs, grid_center.y+hs, grid_center.z+hs };
    float t_min = 0.f, t_max = std::numeric_limits<float>::infinity();
    for(int i = 0; i < 3; i++) {
        float o  = (i==0) ? camera_pos.x     : ((i==1) ? camera_pos.y     : camera_pos.z);
        float d  = (i==0) ? dir_normalized.x  : ((i==1) ? dir_normalized.y  : dir_normalized.z);
        float mn = (i==0) ? gmin.x            : ((i==1) ? gmin.y            : gmin.z);
        float mx = (i==0) ? gmax.x            : ((i==1) ? gmax.y            : gmax.z);
        if(std::fabs(d) < 1e-12f) { if(o < mn || o > mx) return; }
        else {
            float t1=(mn-o)/d, t2=(mx-o)/d;
            float tn=std::fmin(t1,t2), tf=std::fmax(t1,t2);
            if(tn > t_min) t_min = tn;
            if(tf < t_max) t_max = tf;
            if(t_min > t_max) return;
        }
    }
    if(t_min < 0.f) t_min = 0.f;

    // Phase 2: DDA initialisation.
    const Vec3 sw = {
        camera_pos.x + t_min * dir_normalized.x,
        camera_pos.y + t_min * dir_normalized.y,
        camera_pos.z + t_min * dir_normalized.z
    };
    int ix = (int)((sw.x - gmin.x) / voxel_size);
    int iy = (int)((sw.y - gmin.y) / voxel_size);
    int iz = (int)((sw.z - gmin.z) / voxel_size);
    if(ix<0||ix>=N||iy<0||iy>=N||iz<0||iz>=N) return;

    const int step_x = (dir_normalized.x >= 0.f) ? 1 : -1;
    const int step_y = (dir_normalized.y >= 0.f) ? 1 : -1;
    const int step_z = (dir_normalized.z >= 0.f) ? 1 : -1;
    const int nx_x   = ix + (step_x > 0 ? 1 : 0);
    const int nx_y   = iy + (step_y > 0 ? 1 : 0);
    const int nx_z   = iz + (step_z > 0 ? 1 : 0);

    float t_max_x = safe_div(gmin.x + nx_x*voxel_size - camera_pos.x, dir_normalized.x);
    float t_max_y = safe_div(gmin.y + nx_y*voxel_size - camera_pos.y, dir_normalized.y);
    float t_max_z = safe_div(gmin.z + nx_z*voxel_size - camera_pos.z, dir_normalized.z);
    const float t_delta_x = safe_div(voxel_size, std::fabs(dir_normalized.x));
    const float t_delta_y = safe_div(voxel_size, std::fabs(dir_normalized.y));
    const float t_delta_z = safe_div(voxel_size, std::fabs(dir_normalized.z));

    // Cache profile hot values as locals to avoid repeated pointer indirection.
    const float nfs       = prof.near_field_suppression;
    const float ramp_len  = prof.max_useful_distance - prof.min_start_distance;
    const bool  use_ramp  = (nfs > 1e-6f && ramp_len > 1e-3f);
    const float min_dist  = prof.min_start_distance;
    const int   step_limit = (prof.max_steps > 0) ? prof.max_steps : INT_MAX;

    float t_cur   = t_min;
    int   n_steps = 0;

    // Phase 3: DDA traversal with inline accumulation.
    // Accumulate BEFORE advancing (t_cur is the entry distance for the current cell).
    while(t_cur <= t_max && n_steps < step_limit) {

        if(t_cur >= min_dist) {
            float weight = 1.f;
            if(use_ramp) {
                float t = (t_cur - min_dist) / ramp_len;
                if(t > 1.f) t = 1.f;
                weight = (1.f - nfs) + nfs * t;
            }
            const float val = pix_val * weight;
            const int   idx = ix * N * N + iy * N + iz;
            #pragma omp atomic update
            voxel_grid[idx] += val;
        }

        // Advance: step along the axis whose next boundary is nearest.
        if(t_max_x < t_max_y && t_max_x < t_max_z) {
            ix    += step_x;  t_cur = t_max_x;  t_max_x += t_delta_x;
        } else if(t_max_y < t_max_z) {
            iy    += step_y;  t_cur = t_max_y;  t_max_y += t_delta_y;
        } else {
            iz    += step_z;  t_cur = t_max_z;  t_max_z += t_delta_z;
        }
        n_steps++;
        if(ix<0||ix>=N||iy<0||iy>=N||iz<0||iz>=N) break;
    }
}


// ============================================================
// Section 7: Main Pipeline
// ============================================================

int main(int argc, char** argv) {
    if(argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <metadata.json> <image_folder> <output.bin>\n";
        return 1;
    }
    const std::string metadata_path = argv[1];
    const std::string images_folder = argv[2];
    const std::string output_bin    = argv[3];

    std::cout << "[ray_voxel] OMP threads: " << omp_get_max_threads() << "\n";
    auto t_total = HRC::now();

    // -------------------------------------------------------
    // Step 7.1: Load and organise metadata.
    // -------------------------------------------------------
    auto t0 = HRC::now();
    std::vector<FrameInfo> frames = load_metadata(metadata_path);
    if(frames.empty()) { std::cerr << "No frames loaded.\n"; return 1; }
    std::map<int, std::vector<FrameInfo>> frames_by_cam;
    for(const auto& f : frames) frames_by_cam[f.camera_index].push_back(f);
    for(auto& kv : frames_by_cam) {
        auto& v = kv.second;
        std::sort(v.begin(), v.end(), [](const FrameInfo& a, const FrameInfo& b){
            return a.frame_index < b.frame_index;
        });
    }
    std::cout << "[Timing] Metadata:         " << ms_since(t0) << " ms\n";

    // -------------------------------------------------------
    // Step 7.2: Load profile and resolve parameters.
    // -------------------------------------------------------
    t0 = HRC::now();
    Profile prof = load_profile(metadata_path);

    if(prof.center_auto) {
        double sx=0, sy=0, sz=0;
        for(const auto& f : frames){ sx+=f.camera_position.x; sy+=f.camera_position.y; sz+=f.camera_position.z; }
        double n = (double)frames.size();
        prof.grid_center = { float(sx/n), float(sy/n), float(sz/n) };
        std::cout << "[Profile] Auto-centred grid at ("
                  << prof.grid_center.x << "," << prof.grid_center.y
                  << "," << prof.grid_center.z << ")\n";
    }

    // Resolve min_start_distance from rig geometry when set to -1.
    if(prof.min_start_distance < 0.f) {
        std::vector<Vec3> cp;
        for(const auto& kv : frames_by_cam)
            if(!kv.second.empty()) cp.push_back(kv.second[0].camera_position);
        float md = 0;
        for(size_t a=0; a<cp.size(); a++)
            for(size_t b=a+1; b<cp.size(); b++) {
                float dx=cp[a].x-cp[b].x, dy=cp[a].y-cp[b].y, dz=cp[a].z-cp[b].z;
                md = std::fmax(md, std::sqrt(dx*dx+dy*dy+dz*dz));
            }
        prof.min_start_distance = std::fmax(md*0.5f, prof.voxel_size*3.f);
        std::cout << "[Profile] min_start_distance auto: "
                  << prof.min_start_distance << " (rig diameter=" << md << ")\n";
    } else {
        std::cout << "[Profile] min_start_distance: " << prof.min_start_distance << "\n";
    }

    // Resolve max_useful_distance when set to -1.
    if(prof.max_useful_distance < 0.f) {
        float hd = std::sqrt(3.f) * 0.5f * prof.resolution * prof.voxel_size;
        double sx=0, sy=0, sz=0; int np=0;
        for(const auto& kv : frames_by_cam) if(!kv.second.empty()){
            sx+=kv.second[0].camera_position.x;
            sy+=kv.second[0].camera_position.y;
            sz+=kv.second[0].camera_position.z;
            np++;
        }
        if(np > 0) {
            float dx=prof.grid_center.x-float(sx/np);
            float dy=prof.grid_center.y-float(sy/np);
            float dz=prof.grid_center.z-float(sz/np);
            prof.max_useful_distance = std::sqrt(dx*dx+dy*dy+dz*dz) + hd;
        } else {
            prof.max_useful_distance = hd;
        }
        std::cout << "[Profile] max_useful_distance auto: " << prof.max_useful_distance << "\n";
    } else {
        std::cout << "[Profile] max_useful_distance: " << prof.max_useful_distance << "\n";
    }

    std::cout << "[Profile] Grid: " << prof.resolution << "^3"
              << ", voxel_size=" << prof.voxel_size
              << ", image_scale=" << prof.image_scale << "\n";
    std::cout << "[Timing] Profile resolve:  " << ms_since(t0) << " ms\n";

    const int   N           = prof.resolution;
    const float voxel_size  = prof.voxel_size;
    const Vec3  grid_center = prof.grid_center;

    // -------------------------------------------------------
    // Step 7.3: Allocate dense voxel grid.
    //
    // Memory: N^3 * 4 bytes.
    //   N=100 →   4 MB   (calibration target)
    //   N=200 →  32 MB
    //   N=500 → 500 MB   (production; may be tight on $200 embedded hardware)
    //
    // If allocation fails, reduce resolution in profile.json.
    // -------------------------------------------------------
    t0 = HRC::now();
    const long long N_total = (long long)N * N * N;
    std::cout << "[Grid] Allocating dense grid: N=" << N
              << " (" << (N_total * 4 / 1024 / 1024) << " MB)\n";
    std::vector<float> voxel_grid;
    try {
        voxel_grid.assign(N_total, 0.f);
    } catch(const std::bad_alloc&) {
        std::cerr << "FATAL: Cannot allocate " << (N_total * 4 / 1024 / 1024)
                  << " MB for dense grid. Lower resolution in profile.json.\n";
        return 1;
    }
    std::cout << "[Timing] Grid alloc:       " << ms_since(t0) << " ms\n";

    // -------------------------------------------------------
    // Step 7.4: Per-camera motion detection and ray accumulation.
    // -------------------------------------------------------
    auto t_proc = HRC::now();

    for(auto& kv : frames_by_cam) {
        int  cam_id    = kv.first;
        auto& cam_frames = kv.second;

        float cam_threshold = prof.pixel_diff_threshold;
        int   cam_stride    = prof.frame_stride;
        auto ov_it = prof.camera_overrides.find(cam_id);
        if(ov_it != prof.camera_overrides.end()) {
            if(ov_it->second.pixel_diff_threshold >= 0.f) cam_threshold = ov_it->second.pixel_diff_threshold;
            if(ov_it->second.frame_stride > 0)            cam_stride    = ov_it->second.frame_stride;
        }
        if((int)cam_frames.size() < cam_stride + 1) continue;

        std::vector<size_t> pair_starts;
        for(size_t i = 0; i + cam_stride < cam_frames.size(); i += cam_stride)
            pair_starts.push_back(i);

        for(size_t pi = 0; pi < pair_starts.size(); pi++) {
            size_t prev_idx = pair_starts[pi];
            size_t curr_idx = prev_idx + cam_stride;
            std::string prev_path = images_folder + "/" + cam_frames[prev_idx].image_file;
            std::string curr_path = images_folder + "/" + cam_frames[curr_idx].image_file;

            // Load images.
            auto t_load = HRC::now();
            ImageGray prev_img, curr_img;
            if(!load_image_gray(prev_path, prev_img)) {
                std::cerr << "Skipping pair: failed to load " << prev_path << "\n"; continue;
            }
            if(!load_image_gray(curr_path, curr_img)) {
                std::cerr << "Skipping pair: failed to load " << curr_path << "\n"; continue;
            }

            // NEW: Downscale before motion detection.
            // Controlled by profile.json: { "image": { "scale": 0.25 } }
            // 0.25 = quarter linear = 16x fewer pixels = 16x fewer rays to cast.
            // The FOV is unchanged; focal_len is recomputed from the downscaled width.
            if(prof.image_scale < 1.f - 1e-4f) {
                prev_img = downscale_image(prev_img, prof.image_scale);
                curr_img = downscale_image(curr_img, prof.image_scale);
            }
            std::cout << "[Cam " << cam_id << " pair " << pi << "]"
                      << " load+scale: " << ms_since(t_load) << " ms"
                      << " res=" << curr_img.width << "x" << curr_img.height << "\n";

            // Motion detection.
            auto t_motion = HRC::now();
            MotionMask mm = detect_motion(prev_img, curr_img, cam_threshold);
            // Count changed pixels for reporting (not in the hot path).
            long n_changed = 0;
            for(int i = 0; i < mm.width * mm.height; i++) n_changed += mm.changed[i];
            std::cout << "[Cam " << cam_id << " pair " << pi << "]"
                      << " motion detect: " << ms_since(t_motion) << " ms"
                      << " changed=" << n_changed << "\n";

            // Build the camera model for the current frame.
            const FrameInfo& curr_info = cam_frames[curr_idx];
            const Vec3  cam_pos   = curr_info.camera_position;
            const Mat3  cam_rot   = camera_matrix_from_generator_convention(
                                        curr_info.yaw, curr_info.pitch, curr_info.roll);
            const float fov_rad   = deg2rad(curr_info.fov_degrees);
            // focal_len uses the downscaled mm.width — correct because FOV is unchanged.
            const float focal_len = (mm.width * 0.5f) / std::tan(fov_rad * 0.5f);
            const int   W = mm.width, H = mm.height;

            // DDA accumulation — OpenMP parallelizes over image rows.
            // Each thread processes complete rows (schedule dynamic, 16 rows per chunk).
            // cast_ray_and_accumulate() is thread-safe via #pragma omp atomic on the grid.
            auto t_dda = HRC::now();
            #pragma omp parallel for schedule(dynamic, 16)
            for(int v = 0; v < H; v++) {
                for(int u = 0; u < W; u++) {
                    if(!mm.changed[v * W + u]) continue;
                    const float pix_val = mm.diff[v * W + u];
                    if(pix_val < 1e-3f) continue;

                    // Build camera-space ray and rotate to world space.
                    const float rx = float(u) - 0.5f * W;
                    const float ry = -(float(v) - 0.5f * H);  // flip Y: image-down → camera-up
                    const float rz = -focal_len;               // image plane is in front of camera
                    const Vec3 ray_cam   = normalize({rx, ry, rz});
                    const Vec3 ray_world = normalize(mat3_mul_vec3(cam_rot, ray_cam));

                    cast_ray_and_accumulate(
                        cam_pos, ray_world, N, voxel_size, grid_center,
                        pix_val, prof, voxel_grid);
                }
            }
            std::cout << "[Cam " << cam_id << " pair " << pi << "]"
                      << " DDA: " << ms_since(t_dda) << " ms\n";
        }
    }
    std::cout << "[Timing] All cameras:      " << ms_since(t_proc) << " ms\n";

    // -------------------------------------------------------
    // Step 7.5: Write sparse binary output.
    //
    // Iterates the dense grid and writes only non-zero entries.
    // Output format is identical to v1 — backward-compatible with all readers.
    //
    // Sparse binary format:
    //   Bytes  0– 3:  int   N           (grid resolution)
    //   Bytes  4– 7:  float voxel_size
    //   Bytes  8–11:  int   count       (number of non-zero entries)
    //   Bytes 12–...: [int flat_index, float value] × count  (8 bytes each)
    //
    //   flat_index = ix*N*N + iy*N + iz
    //   Decode: ix = idx/(N*N),  iy = (idx/N)%N,  iz = idx%N
    // -------------------------------------------------------
    {
        auto t_out = HRC::now();
        std::ofstream ofs(output_bin, std::ios::binary);
        if(!ofs) { std::cerr << "Cannot open output: " << output_bin << "\n"; return 1; }

        // Single pass: count then write (avoids building an intermediate vector).
        int count = 0;
        for(long long i = 0; i < N_total; i++) count += (voxel_grid[i] > 0.f) ? 1 : 0;

        ofs.write(reinterpret_cast<const char*>(&N),          sizeof(int));
        ofs.write(reinterpret_cast<const char*>(&voxel_size), sizeof(float));
        ofs.write(reinterpret_cast<const char*>(&count),      sizeof(int));

        for(long long i = 0; i < N_total; i++) {
            if(voxel_grid[i] > 0.f) {
                int   idx = static_cast<int>(i);
                float val = voxel_grid[i];
                ofs.write(reinterpret_cast<const char*>(&idx), sizeof(int));
                ofs.write(reinterpret_cast<const char*>(&val), sizeof(float));
            }
        }
        ofs.close();
        std::cout << "[Output] " << output_bin
                  << "  " << count << " active voxels"
                  << "  " << (12 + count * 8) / 1024 << " KB\n";
        std::cout << "[Timing] Output write:     " << ms_since(t_out) << " ms\n";
    }

    std::cout << "[Timing] Total:            " << ms_since(t_total) << " ms\n";
    return 0;
}