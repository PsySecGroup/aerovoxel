/**
 * @file ray_voxel.cpp
 * @brief Multi-camera motion-based voxel reconstruction via 3D DDA ray casting.
 *
 * ## Overview
 * This program reconstructs a 3D voxel occupancy grid from a set of
 * multi-camera grayscale image sequences. The pipeline is:
 *
 *   1. Parse a JSON metadata file describing each frame's camera pose and image path.
 *   2. Load consecutive image pairs per camera and detect per-pixel motion via
 *      absolute difference thresholding.
 *   3. For every pixel where motion is detected, cast a ray from the camera into a
 *      shared 3D voxel grid using a Digital Differential Analyzer (DDA) traversal.
 *   4. Accumulate motion-signal energy along each ray into the voxel grid. Voxels
 *      that are consistently "seen" by multiple rays across many cameras and frames
 *      accumulate higher values, forming a volumetric reconstruction of moving objects.
 *   5. Write the final voxel grid to a flat binary (.bin) file for downstream
 *      visualization or analysis.
 *
 * ## Coordinate System
 * - World space is right-handed: X = right, Y = up, Z = forward (away from cameras).
 * - Camera space: X = right, Y = up, Z = into the screen (negative forward).
 * - Euler angles are applied as Rz(yaw) * Ry(roll) * Rx(pitch), transforming
 *   a camera-space ray direction into world space.
 *
 * ## Usage
 *   ray_voxel <metadata.json> <image_folder> <output.bin>
 *
 *   - metadata.json : JSON array of frame descriptors (see load_metadata()).
 *   - image_folder  : Directory containing the image files referenced in metadata.
 *   - output.bin    : Binary output file; starts with (int N, float voxel_size),
 *                     followed by N^3 floats in [ix][iy][iz] order.
 *
 * ## Key Tuning Parameters (in main())
 *   - N              : Grid resolution (N x N x N voxels), default 500.
 *   - voxel_size     : Side length of each cubic voxel in world units, default 6.0.
 *   - grid_center    : World-space center of the voxel volume.
 *   - motion_threshold : Absolute pixel-difference threshold to flag motion.
 *   - alpha          : Distance-attenuation coefficient (currently unused in
 *                      accumulation but computed; see note in accumulation loop).
 *
 * ## Dependencies
 *   - nlohmann/json  : Header-only JSON parser.
 *   - stb_image      : Header-only image loader (force-included via STB_IMAGE_IMPLEMENTATION).
 */

#include <iostream>
#include <fstream>
#include <cmath>
#include <limits>
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <algorithm>

// nlohmann/json: single-header JSON library used to parse the metadata file.
#include "nlohmann/json.hpp"

// stb_image: single-header image loader. STB_IMAGE_IMPLEMENTATION must be
// defined exactly once across the project to instantiate the implementation.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Bring nlohmann::json into scope as the shorter alias `json`.
using json = nlohmann::json;


// ============================================================
// Section 1: Core Data Structures
// ============================================================

/**
 * @brief A minimal 3-component float vector for positions and directions.
 *
 * Used throughout for world-space camera positions, ray directions,
 * grid bounds, and intermediate calculations.
 */
struct Vec3 {
    float x, y, z;
};

/**
 * @brief A row-major 3x3 float matrix.
 *
 * Element (row, col) is stored at m[row*3 + col].
 * Used exclusively to represent camera rotation matrices built from
 * Euler angles (yaw, pitch, roll).
 */
struct Mat3 {
    float m[9]; ///< Row-major storage: m[0..2] = row 0, m[3..5] = row 1, m[6..8] = row 2.
};

/**
 * @brief Metadata describing a single camera frame.
 *
 * Each entry in the input JSON array is parsed into one FrameInfo.
 * The pipeline groups FrameInfo records by camera_index and sorts them
 * by frame_index to establish the temporal sequence for motion detection.
 */
struct FrameInfo {
    int   camera_index;      ///< Which camera captured this frame (0-based).
    int   frame_index;       ///< Temporal position within this camera's sequence (0-based).
    Vec3  camera_position;   ///< World-space position of the camera optical centre.
    float yaw;               ///< Rotation around Z-axis in degrees (azimuth).
    float pitch;             ///< Rotation around X-axis in degrees (elevation).
    float roll;              ///< Rotation around Y-axis in degrees (bank/tilt).
    float fov_degrees;       ///< Horizontal field-of-view in degrees.
    std::string image_file;  ///< Filename (not full path) of the associated image.
};


// ============================================================
// Section 2: Math Helpers
// ============================================================

/**
 * @brief Convert degrees to radians.
 * @param deg Angle in degrees.
 * @return Equivalent angle in radians.
 */
static inline float deg2rad(float deg) {
    return deg * 3.14159265358979323846f / 180.0f;
}

/**
 * @brief Normalize a Vec3 to unit length.
 *
 * If the input vector has a near-zero magnitude (< 1e-12), the zero vector
 * is returned to avoid division by zero. This case should not arise in
 * normal ray-casting usage but is guarded defensively.
 *
 * @param v Input vector.
 * @return Unit vector in the same direction, or {0,0,0} if v is degenerate.
 */
static inline Vec3 normalize(const Vec3 &v) {
    float len = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    if(len < 1e-12f) {
        return {0.f, 0.f, 0.f};
    }
    return { v.x/len, v.y/len, v.z/len };
}

/**
 * @brief Multiply a 3x3 matrix by a column vector.
 *
 * Computes r = M * v using standard row-major matrix-vector multiplication.
 * This is used to rotate camera-space ray directions into world space.
 *
 * @param M Row-major 3x3 rotation matrix.
 * @param v Column vector to transform.
 * @return Transformed vector M*v.
 */
static inline Vec3 mat3_mul_vec3(const Mat3 &M, const Vec3 &v) {
    Vec3 r;
    r.x = M.m[0]*v.x + M.m[1]*v.y + M.m[2]*v.z;
    r.y = M.m[3]*v.x + M.m[4]*v.y + M.m[5]*v.z;
    r.z = M.m[6]*v.x + M.m[7]*v.y + M.m[8]*v.z;
    return r;
}


// ============================================================
// Section 3: Euler Angles to Rotation Matrix
// ============================================================

/**
 * @brief Build a combined rotation matrix from yaw, pitch, and roll angles.
 *
 * Constructs the composite rotation R = Rz(yaw) * Ry(roll) * Rx(pitch),
 * applied in that order (right-to-left in matrix form). This matches
 * a common aerospace / navigation convention where:
 *   - Yaw   (psi)   : rotation about the Z-axis (heading).
 *   - Pitch (theta) : rotation about the X-axis (nose up/down).
 *   - Roll  (phi)   : rotation about the Y-axis (bank left/right).
 *
 * The resulting matrix transforms a vector from camera/body space
 * into world space. To go from world to camera space, transpose the result
 * (since rotation matrices are orthogonal).
 *
 * Component sub-matrices (row-major, right-handed):
 * @code
 *   Rz(yaw)   = [[ cy, -sy,  0 ],   // rotation in XY plane
 *                [ sy,  cy,  0 ],
 *                [  0,   0,  1 ]]
 *
 *   Ry(roll)  = [[ cr,   0, sr ],   // rotation in XZ plane
 *                [  0,   1,  0 ],
 *                [-sr,   0, cr ]]
 *
 *   Rx(pitch) = [[  1,   0,  0 ],   // rotation in YZ plane
 *                [  0,  cp, -sp],
 *                [  0,  sp,  cp]]
 * @endcode
 *
 * @param yaw_deg   Yaw angle in degrees.
 * @param pitch_deg Pitch angle in degrees.
 * @param roll_deg  Roll angle in degrees.
 * @return 3x3 rotation matrix (row-major) representing the combined rotation.
 */
Mat3 rotation_matrix_yaw_pitch_roll(float yaw_deg, float pitch_deg, float roll_deg) {
    float y = deg2rad(yaw_deg);
    float p = deg2rad(pitch_deg);
    float r = deg2rad(roll_deg);

    // --- Rz: Yaw (rotation about Z-axis) ---
    float cy = std::cos(y), sy = std::sin(y);
    float Rz[9] = {
        cy, -sy, 0.f,
        sy,  cy, 0.f,
        0.f, 0.f, 1.f
    };

    // --- Ry: Roll (rotation about Y-axis) ---
    float cr = std::cos(r), sr = std::sin(r);
    float Ry[9] = {
        cr,  0.f, sr,
        0.f, 1.f, 0.f,
        -sr, 0.f, cr
    };

    // --- Rx: Pitch (rotation about X-axis) ---
    float cp = std::cos(p), sp = std::sin(p);
    float Rx[9] = {
        1.f,  0.f,  0.f,
        0.f,  cp,  -sp,
        0.f,  sp,   cp
    };

    /**
     * Local lambda to multiply two 3x3 row-major matrices: C = A * B.
     * Defined here to avoid polluting the outer namespace since it's only
     * needed during this rotation-matrix construction.
     */
    auto matmul3x3 = [&](const float A[9], const float B[9], float C[9]) {
        for(int row = 0; row < 3; ++row) {
            for(int col = 0; col < 3; ++col) {
                C[row*3+col] =
                    A[row*3+0]*B[0*3+col] +
                    A[row*3+1]*B[1*3+col] +
                    A[row*3+2]*B[2*3+col];
            }
        }
    };

    // Compose: R = (Rz * Ry) * Rx
    float Rtemp[9], Rfinal[9];
    matmul3x3(Rz, Ry, Rtemp);     // First: Rz * Ry
    matmul3x3(Rtemp, Rx, Rfinal); // Then:  (Rz*Ry) * Rx

    Mat3 out;
    for(int i = 0; i < 9; i++) {
        out.m[i] = Rfinal[i];
    }
    return out;
}


// ============================================================
// Section 3b: Generator-Convention Camera Matrix
// ============================================================

/**
 * @brief Build a camera-to-world rotation matrix matching generator.py's convention.
 *
 * ## Why this exists
 * The original rotation_matrix_yaw_pitch_roll() uses an aerospace Euler convention
 * where the camera's base forward direction is -Z in world space, and Rz(yaw)
 * rotates within the XY plane. The critical flaw: Rz applied to (0,0,-1) leaves
 * the Z component unchanged — yaw never actually redirects the camera forward axis,
 * only the fan-out direction of off-centre pixels.
 *
 * generator.py computes camera direction as:
 * @code
 *   dx = cos(pitch) * sin(yaw)   // X: lateral
 *   dy = cos(pitch) * cos(yaw)   // Y: depth (forward in generator space)
 *   dz = sin(pitch)              // Z: up
 * @endcode
 *
 * This convention uses Y as the depth/forward axis and Z as the vertical axis,
 * matching a standard surveying / GIS frame. Yaw=0 looks in +Y, yaw=90 looks in +X.
 *
 * ## How this matrix is built
 * Given the generator forward vector F, we construct a full orthonormal camera basis:
 * @code
 *   world_up = (0, 0, 1)             // Z is up in generator space
 *   right    = normalize(F x up)     // camera +X in world space
 *   local_up = normalize(right x F)  // camera +Y in world space (perpendicular to F)
 * @endcode
 *
 * The rotation matrix R (row-major, camera-space → world-space) satisfies:
 * @code
 *   R * (1, 0,  0) = right      // camera right  → world right
 *   R * (0, 1,  0) = local_up   // camera up     → world up
 *   R * (0, 0, -1) = F          // camera forward → world forward
 * @endcode
 *
 * The row layout therefore is:
 * @code
 *   row 0: [right.x,    local_up.x,  -F.x]
 *   row 1: [right.y,    local_up.y,  -F.y]
 *   row 2: [right.z,    local_up.z,  -F.z]
 * @endcode
 *
 * @param yaw_deg   Azimuth angle in degrees. 0 = +Y, 90 = +X, 180 = -Y, 270 = -X.
 * @param pitch_deg Elevation angle in degrees. 0 = horizontal, 90 = straight up.
 * @param roll_deg  Bank angle in degrees (rotation around the forward axis).
 * @return Row-major 3x3 camera-to-world rotation matrix.
 */
Mat3 camera_matrix_from_generator_convention(float yaw_deg, float pitch_deg, float roll_deg) {
    float y = deg2rad(yaw_deg);
    float p = deg2rad(pitch_deg);

    // Forward direction as computed by generator.py (Y-forward, Z-up frame).
    Vec3 F = {
        std::cos(p) * std::sin(y),   // X component
        std::cos(p) * std::cos(y),   // Y component (depth)
        std::sin(p)                   // Z component (up)
    };
    F = normalize(F);

    // World up axis in generator space.
    Vec3 world_up = {0.f, 0.f, 1.f};

    // Right = F x world_up. Normalise to handle near-vertical cameras gracefully.
    Vec3 right = normalize({
        F.y * world_up.z - F.z * world_up.y,
        F.z * world_up.x - F.x * world_up.z,
        F.x * world_up.y - F.y * world_up.x
    });

    // If F is nearly parallel to world_up (camera pointing straight up or down),
    // fall back to using +X as the reference for right to avoid a degenerate basis.
    float dot_F_up = F.x*world_up.x + F.y*world_up.y + F.z*world_up.z;
    if(std::fabs(dot_F_up) > 0.999f) {
        right = normalize({1.f, 0.f, 0.f});
    }

    // Local up = right x F (gives the camera's +Y axis in world space).
    Vec3 local_up = normalize({
        right.y * F.z - right.z * F.y,
        right.z * F.x - right.x * F.z,
        right.x * F.y - right.y * F.x
    });

    // Apply roll as a rotation around F (bank the camera).
    if(std::fabs(roll_deg) > 1e-4f) {
        float r = deg2rad(roll_deg);
        float cr = std::cos(r), sr = std::sin(r);
        Vec3 new_right    = { cr*right.x + sr*local_up.x,
                              cr*right.y + sr*local_up.y,
                              cr*right.z + sr*local_up.z };
        Vec3 new_local_up = {-sr*right.x + cr*local_up.x,
                             -sr*right.y + cr*local_up.y,
                             -sr*right.z + cr*local_up.z };
        right    = new_right;
        local_up = new_local_up;
    }

    // Assemble row-major rotation matrix.
    // Row i = [right[i], local_up[i], -F[i]]
    Mat3 M;
    M.m[0] = right.x;    M.m[1] = local_up.x;  M.m[2] = -F.x;
    M.m[3] = right.y;    M.m[4] = local_up.y;  M.m[5] = -F.y;
    M.m[6] = right.z;    M.m[7] = local_up.z;  M.m[8] = -F.z;
    return M;
}


// ============================================================
// Section 3c: Profile — Runtime Calibration Parameters
// ============================================================

/**
 * @brief All tunable parameters for a single scene run.
 *
 * Values are loaded from an optional `profile.json` file located in the
 * same directory as the metadata file. Any field absent from the JSON
 * falls back to the default value shown here, so a profile only needs to
 * contain the parameters that differ from the defaults.
 *
 * ## profile.json schema
 * @code
 * {
 *   "grid": {
 *     "resolution":  500,       // voxels per axis (N x N x N grid)
 *     "voxel_size":  6.0,       // world-unit side length of one voxel
 *     "center": "auto"          // or [x, y, z] — world-space grid centre
 *   },
 *   "motion": {
 *     "pixel_diff_threshold":    2.0,  // global per-pixel motion threshold
 *     "attenuation_coefficient": 0.1   // distance-attenuation factor (future use)
 *   },
 *   "ray": {
 *     "max_steps":    500,   // maximum DDA steps per ray (0 = unlimited)
 *     "frame_stride": 1      // diff frame[i] vs frame[i+stride]; advance by stride
 *   },
 *   "clustering": {
 *     "min_corroborating_cameras": 2,    // cameras that must agree on a voxel
 *     "noise_floor_percentile":    0.90, // discard bottom X% by value before clustering
 *     "merge_radius":             -1     // cluster merge radius (-1 = 3x voxel_size)
 *   },
 *   "cameras": {
 *     "0": { "pixel_diff_threshold": 5.0 },
 *     "1": { "pixel_diff_threshold": 1.5, "frame_stride": 2 }
 *   }
 * }
 * @endcode
 */
struct Profile {
    // --- grid ---
    int   resolution  = 500;
    float voxel_size  = 6.f;
    bool  center_auto = true;
    Vec3  grid_center = {0.f, 0.f, 500.f};

    // --- motion ---
    float pixel_diff_threshold    = 2.0f;
    float attenuation_coefficient = 0.1f;

    // --- ray ---
    int max_steps    = 0;   ///< 0 = unlimited
    int frame_stride = 1;

    // --- clustering ---
    int   min_corroborating_cameras = 2;
    float noise_floor_percentile    = 0.90f;
    float merge_radius              = -1.f; ///< -1 = 3 x voxel_size

    // --- per-camera overrides (camera_index -> overridden values) ---
    struct CameraOverride {
        float pixel_diff_threshold = -1.f; ///< -1 = use global default
        int   frame_stride         = -1;   ///< -1 = use global default
    };
    std::map<int, CameraOverride> camera_overrides;
};

/**
 * @brief Load a Profile from a profile.json file, with full fallback to defaults.
 *
 * Looks for `profile.json` in the same directory as `metadata_path`. If the
 * file does not exist, or if individual keys are absent, the corresponding
 * Profile fields keep their default values — making every field optional.
 *
 * @param metadata_path Path to the metadata JSON (used to locate the profile).
 * @return Populated Profile; all-defaults if no profile.json is found.
 */
Profile load_profile(const std::string &metadata_path) {
    Profile prof;

    // Derive profile path: same directory as metadata, named "profile.json".
    std::string dir = metadata_path;
    auto slash = dir.find_last_of("/\\");
    if(slash != std::string::npos) dir = dir.substr(0, slash);
    else                            dir = ".";
    std::string profile_path = dir + "/profile.json";

    std::ifstream ifs(profile_path);
    if(!ifs.is_open()) {
        std::cout << "[Profile] No profile.json found at " << profile_path
                  << " — using defaults.\n";
        return prof;
    }

    json j;
    try { ifs >> j; }
    catch(const std::exception &e) {
        std::cerr << "[Profile] Failed to parse " << profile_path
                  << ": " << e.what() << " — using defaults.\n";
        return prof;
    }
    std::cout << "[Profile] Loaded " << profile_path << "\n";

    // --- grid ---
    if(j.contains("grid")) {
        auto &g = j["grid"];
        if(g.contains("resolution")) prof.resolution = g["resolution"].get<int>();
        if(g.contains("voxel_size")) prof.voxel_size  = g["voxel_size"].get<float>();
        if(g.contains("center")) {
            auto &c = g["center"];
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

    // --- motion ---
    if(j.contains("motion")) {
        auto &m = j["motion"];
        if(m.contains("pixel_diff_threshold"))
            prof.pixel_diff_threshold = m["pixel_diff_threshold"].get<float>();
        if(m.contains("attenuation_coefficient"))
            prof.attenuation_coefficient = m["attenuation_coefficient"].get<float>();
    }

    // --- ray ---
    if(j.contains("ray")) {
        auto &r = j["ray"];
        if(r.contains("max_steps"))    prof.max_steps    = r["max_steps"].get<int>();
        if(r.contains("frame_stride")) prof.frame_stride = r["frame_stride"].get<int>();
    }

    // --- clustering ---
    if(j.contains("clustering")) {
        auto &cl = j["clustering"];
        if(cl.contains("min_corroborating_cameras"))
            prof.min_corroborating_cameras = cl["min_corroborating_cameras"].get<int>();
        if(cl.contains("noise_floor_percentile"))
            prof.noise_floor_percentile = cl["noise_floor_percentile"].get<float>();
        if(cl.contains("merge_radius"))
            prof.merge_radius = cl["merge_radius"].get<float>();
    }

    // --- per-camera overrides ---
    if(j.contains("cameras") && j["cameras"].is_object()) {
        for(auto &[key, val] : j["cameras"].items()) {
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
// Section 4: JSON Metadata Loading
// ============================================================

/**
 * @brief Parse the JSON metadata file and return a list of frame descriptors.
 *
 * The JSON file must be a top-level array where each element is an object
 * with the following fields (all optional with sensible defaults):
 * @code
 * [
 *   {
 *     "camera_index":    0,
 *     "frame_index":     0,
 *     "camera_position": [x, y, z],
 *     "yaw":             0.0,
 *     "pitch":           0.0,
 *     "roll":            0.0,
 *     "fov_degrees":     60.0,
 *     "image_file":      "cam0_frame0.png"
 *   },
 *   ...
 * ]
 * @endcode
 *
 * Fields not present in an entry receive the following defaults:
 *   - camera_index:    0
 *   - frame_index:     0
 *   - yaw/pitch/roll:  0.0
 *   - fov_degrees:     60.0
 *   - image_file:      ""  (will cause a load error downstream)
 *   - camera_position: {0, 0, 0} (if array is missing or has < 3 elements)
 *
 * @param json_path Filesystem path to the metadata JSON file.
 * @return Vector of FrameInfo; empty if the file cannot be opened or parsed.
 */
std::vector<FrameInfo> load_metadata(const std::string &json_path) {
    std::vector<FrameInfo> frames;

    std::ifstream ifs(json_path);
    if(!ifs.is_open()) {
        std::cerr << "ERROR: Cannot open " << json_path << std::endl;
        return frames;
    }

    json j;
    ifs >> j;
    if(!j.is_array()) {
        std::cerr << "ERROR: JSON top level is not an array.\n";
        return frames;
    }

    for(const auto &entry : j) {
        FrameInfo fi;
        fi.camera_index   = entry.value("camera_index",  0);
        fi.frame_index    = entry.value("frame_index",   0);
        fi.yaw            = entry.value("yaw",           0.f);
        fi.pitch          = entry.value("pitch",         0.f);
        fi.roll           = entry.value("roll",          0.f);
        fi.fov_degrees    = entry.value("fov_degrees",   60.f);
        fi.image_file     = entry.value("image_file",    "");

        // camera_position is stored as a JSON array [x, y, z].
        // We check both that the key exists and that it is an array with >= 3 elements.
        if(entry.contains("camera_position") && entry["camera_position"].is_array()) {
            auto arr = entry["camera_position"];
            if(arr.size() >= 3) {
                fi.camera_position.x = arr[0].get<float>();
                fi.camera_position.y = arr[1].get<float>();
                fi.camera_position.z = arr[2].get<float>();
            }
        }

        frames.push_back(fi);
    }

    return frames;
}


// ============================================================
// Section 5: Image Loading and Motion Detection
// ============================================================

/**
 * @brief A single-channel (grayscale) floating-point image.
 *
 * Pixel (u, v) is stored at pixels[v * width + u], where u is the column
 * (horizontal, left-to-right) and v is the row (vertical, top-to-bottom).
 * Values are in the range [0, 255] after loading, matching 8-bit source data.
 */
struct ImageGray {
    int width;                  ///< Image width in pixels.
    int height;                 ///< Image height in pixels.
    std::vector<float> pixels;  ///< Row-major pixel intensities in [0, 255].
};

#include <random>  // std::mt19937, std::uniform_real_distribution

/**
 * @brief Load an image file as a grayscale float image with dithering noise.
 *
 * Uses stb_image to decode the file into 8-bit single-channel (grayscale)
 * data. Each pixel is converted to float and perturbed by a small uniform
 * random value in [-1, +1] before being clamped back to [0, 255].
 *
 * ### Why add noise?
 * Low-quality or heavily compressed images can produce flat regions where
 * consecutive frames have identical pixel values, leading to zero motion
 * signal even when slight real motion exists. The added noise acts as a
 * dithering layer that breaks up quantization artifacts and prevents the
 * absolute-difference detector from being fooled by encoding flatness.
 * The noise magnitude (±1 out of 255) is sub-threshold relative to the
 * default motion_threshold of 2.0, so it will not create false positives
 * on its own.
 *
 * @param img_path Path to the image file (PNG, JPG, BMP, etc.).
 * @param[out] out Populated with width, height, and noisy pixel data on success.
 * @return true if the image was successfully loaded; false otherwise.
 */
bool load_image_gray(const std::string &img_path, ImageGray &out) {
    int w, h, channels;

    // Request exactly 1 output channel (grayscale) regardless of source format.
    unsigned char* data = stbi_load(img_path.c_str(), &w, &h, &channels, 1);
    if(!data) {
        std::cerr << "Failed to load image: " << img_path << std::endl;
        return false;
    }

    out.width  = w;
    out.height = h;
    out.pixels.resize(w * h);

    // Use a thread-local (static) Mersenne Twister seeded from the hardware
    // entropy source. Static ensures one engine is reused across calls rather
    // than re-seeding on every image load, which would degrade randomness.
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> noise_dist(-1.0f, 1.0f);

    for(int i = 0; i < w * h; i++) {
        float val = static_cast<float>(data[i]);  // Convert 8-bit [0,255] to float.
        val += noise_dist(gen);                    // Add sub-threshold dithering noise.
        // Hard-clamp to valid intensity range.
        if(val < 0.0f)   val = 0.0f;
        if(val > 255.0f) val = 255.0f;
        out.pixels[i] = val;
    }

    stbi_image_free(data);
    return true;
}

/**
 * @brief Per-pixel motion detection result between two consecutive frames.
 *
 * Stores both the binary change mask and the raw absolute difference values.
 * The difference values are used directly as the motion-signal strength when
 * accumulating into the voxel grid: brighter motion = stronger evidence of
 * an object at that location.
 */
struct MotionMask {
    int width;                   ///< Image width (must match both input frames).
    int height;                  ///< Image height (must match both input frames).
    std::vector<bool>  changed;  ///< True at pixel (u,v) if |prev - next| > threshold.
    std::vector<float> diff;     ///< Raw absolute difference at each pixel [0, 255].
};

/**
 * @brief Compute a per-pixel motion mask between two grayscale frames.
 *
 * For each pixel i, computes diff[i] = |prev.pixels[i] - next.pixels[i]|.
 * changed[i] is set to true when diff[i] exceeds the threshold. Only changed
 * pixels will have rays cast through the voxel grid.
 *
 * Both images must have the same dimensions. If they differ, an empty mask
 * (width = height = 0) is returned and an error is printed to stderr.
 *
 * @param prev      The earlier frame for this camera.
 * @param next      The later frame for this camera.
 * @param threshold Minimum absolute pixel difference to count as motion.
 *                  Larger values suppress noise-induced false positives but
 *                  may miss slow or subtle movements. Default in main() is 2.0.
 * @return MotionMask populated with per-pixel difference and change flags.
 */
MotionMask detect_motion(const ImageGray &prev, const ImageGray &next, float threshold) {
    MotionMask mm;

    if(prev.width != next.width || prev.height != next.height) {
        std::cerr << "Images differ in size. Can't do motion detection!\n";
        mm.width  = 0;
        mm.height = 0;
        return mm;
    }

    mm.width   = prev.width;
    mm.height  = prev.height;
    mm.changed.resize(mm.width * mm.height, false);
    mm.diff.resize(mm.width * mm.height, 0.f);

    for(int i = 0; i < mm.width * mm.height; i++) {
        float d      = std::fabs(prev.pixels[i] - next.pixels[i]);
        mm.diff[i]   = d;
        mm.changed[i] = (d > threshold);
    }

    return mm;
}


// ============================================================
// Section 6: 3D DDA Voxel Ray Casting
// ============================================================

/**
 * @brief A single voxel cell visited during a ray traversal.
 *
 * Produced by cast_ray_into_grid() for each grid cell the ray passes through,
 * in traversal order from the ray origin. The step_count allows callers to
 * implement early termination or depth-based weighting.
 */
struct RayStep {
    int   ix;          ///< Voxel X-index (0-based, within [0, N-1]).
    int   iy;          ///< Voxel Y-index (0-based, within [0, N-1]).
    int   iz;          ///< Voxel Z-index (0-based, within [0, N-1]).
    int   step_count;  ///< How many DDA steps have been taken to reach this cell (0-based).
    float distance;    ///< Parametric distance t along the ray at which this cell was entered.
};

/**
 * @brief Divide two floats, returning +infinity when the denominator is near zero.
 *
 * Used in the DDA to compute t_max and t_delta values for ray components
 * that are nearly axis-aligned. Returning infinity naturally causes those
 * components to never become the "smallest" t-value and thus never drive
 * a step, which is the correct DDA behaviour for rays parallel to a plane.
 *
 * @param num Numerator.
 * @param den Denominator.
 * @return num/den, or +infinity if |den| < 1e-12.
 */
static inline float safe_div(float num, float den) {
    if(std::fabs(den) < 1e-12f) {
        return std::numeric_limits<float>::infinity();
    }
    return num / den;
}

/**
 * @brief Trace a ray through a cubic voxel grid using the 3D DDA algorithm.
 *
 * ## Algorithm Overview
 * This implements Amanatides & Woo's "A Fast Voxel Traversal Algorithm"
 * (1987), which is the standard for efficient grid ray casting. It proceeds
 * in two phases:
 *
 * ### Phase 1 – Ray-Box Intersection (slab method)
 * The grid is treated as an axis-aligned bounding box (AABB). The ray is
 * intersected with each pair of axis-aligned slabs using:
 * @code
 *   t1 = (slab_min - origin) / direction
 *   t2 = (slab_max - origin) / direction
 *   t_entry = max(t_entry, min(t1, t2))
 *   t_exit  = min(t_exit,  max(t1, t2))
 * @endcode
 * If t_entry > t_exit, the ray misses the grid entirely and an empty list
 * is returned. If t_entry < 0, the camera is inside the grid and we clamp
 * t_entry to 0.
 *
 * ### Phase 2 – DDA Traversal
 * Starting from the entry voxel, the algorithm maintains three "next
 * boundary" distances (t_max_x, t_max_y, t_max_z), one per axis. At each
 * step it advances into the cell whose nearest boundary is smallest:
 * @code
 *   if   t_max_x < t_max_y && t_max_x < t_max_z  →  step in X
 *   elif t_max_y < t_max_z                         →  step in Y
 *   else                                            →  step in Z
 * @endcode
 * This guarantees that every voxel the ray passes through is visited exactly
 * once, in entry order, with no gaps or duplicates.
 *
 * ## Grid Layout
 * The grid is an N×N×N cube of axis-aligned voxels, each of side `voxel_size`
 * world units, centred at `grid_center`. Voxel (ix, iy, iz) occupies the
 * world-space range:
 * @code
 *   X: [grid_min.x + ix*voxel_size,   grid_min.x + (ix+1)*voxel_size]
 *   Y: [grid_min.y + iy*voxel_size,   grid_min.y + (iy+1)*voxel_size]
 *   Z: [grid_min.z + iz*voxel_size,   grid_min.z + (iz+1)*voxel_size]
 * @endcode
 *
 * @param camera_pos     World-space position of the camera (ray origin).
 * @param dir_normalized Pre-normalized world-space ray direction (unit length).
 * @param N              Grid resolution (number of voxels per axis).
 * @param voxel_size     Side length of one voxel in world units.
 * @param grid_center    World-space centre of the entire voxel volume.
 * @return Ordered list of RayStep, one per traversed voxel.
 *         Empty if the ray misses the grid or dir_normalized is degenerate.
 */
std::vector<RayStep> cast_ray_into_grid(
    const Vec3 &camera_pos,
    const Vec3 &dir_normalized,
    int          N,
    float        voxel_size,
    const Vec3  &grid_center)
{
    std::vector<RayStep> steps;
    steps.reserve(64); // Pre-allocate for a typical ray traversal length.

    // Compute AABB bounds from grid centre and total extent.
    float half_size = 0.5f * (N * voxel_size);
    Vec3 grid_min = { grid_center.x - half_size,
                      grid_center.y - half_size,
                      grid_center.z - half_size };
    Vec3 grid_max = { grid_center.x + half_size,
                      grid_center.y + half_size,
                      grid_center.z + half_size };

    // t_min / t_max represent the parametric [entry, exit] interval of the
    // ray within the grid AABB.  They start as [0, +inf] and are progressively
    // narrowed by each axis's slab intersection.
    float t_min = 0.f;
    float t_max = std::numeric_limits<float>::infinity();

    // --- Phase 1: Slab-method AABB intersection ---
    // Iterate once per axis (i=0→X, i=1→Y, i=2→Z).
    for(int i = 0; i < 3; i++) {
        // Extract per-axis components using index; avoids code duplication.
        float origin = (i==0)? camera_pos.x     : ((i==1)? camera_pos.y     : camera_pos.z);
        float d      = (i==0)? dir_normalized.x  : ((i==1)? dir_normalized.y  : dir_normalized.z);
        float mn     = (i==0)? grid_min.x        : ((i==1)? grid_min.y        : grid_min.z);
        float mx     = (i==0)? grid_max.x        : ((i==1)? grid_max.y        : grid_max.z);

        if(std::fabs(d) < 1e-12f) {
            // Ray is parallel to this slab pair; check if origin is inside.
            // If outside, the ray can never intersect the grid.
            if(origin < mn || origin > mx) {
                return steps; // Miss — return empty.
            }
            // Inside: this axis doesn't constrain t_min/t_max, continue.
        } else {
            // Compute parametric distances to both slab boundaries.
            float t1 = (mn - origin) / d;
            float t2 = (mx - origin) / d;
            // Ensure t1 is the nearer boundary.
            float t_near = std::fmin(t1, t2);
            float t_far  = std::fmax(t1, t2);
            // Narrow the valid interval.
            if(t_near > t_min) t_min = t_near;
            if(t_far  < t_max) t_max = t_far;
            // If the interval is empty the ray misses the grid.
            if(t_min > t_max) {
                return steps; // Miss — return empty.
            }
        }
    }

    // If the full grid is behind the camera, clamp to start at the camera.
    if(t_min < 0.f) t_min = 0.f;

    // --- Phase 2: DDA Traversal ---

    // Compute the world-space entry point and convert to fractional voxel coords.
    Vec3 start_world = {
        camera_pos.x + t_min * dir_normalized.x,
        camera_pos.y + t_min * dir_normalized.y,
        camera_pos.z + t_min * dir_normalized.z
    };
    float fx = (start_world.x - grid_min.x) / voxel_size;
    float fy = (start_world.y - grid_min.y) / voxel_size;
    float fz = (start_world.z - grid_min.z) / voxel_size;

    // Integer voxel indices of the starting cell.
    int ix = int(fx);
    int iy = int(fy);
    int iz = int(fz);

    // Guard: if the computed entry cell is somehow out of bounds, abort.
    if(ix < 0 || ix >= N || iy < 0 || iy >= N || iz < 0 || iz >= N) {
        return steps;
    }

    // Step direction per axis: +1 if ray moves in positive axis direction, -1 otherwise.
    int step_x = (dir_normalized.x >= 0.f) ? 1 : -1;
    int step_y = (dir_normalized.y >= 0.f) ? 1 : -1;
    int step_z = (dir_normalized.z >= 0.f) ? 1 : -1;

    // Lambdas to convert an integer voxel boundary index to its world-space coordinate.
    auto boundary_in_world_x = [&](int i_x) { return grid_min.x + i_x * voxel_size; };
    auto boundary_in_world_y = [&](int i_y) { return grid_min.y + i_y * voxel_size; };
    auto boundary_in_world_z = [&](int i_z) { return grid_min.z + i_z * voxel_size; };

    // Index of the next voxel boundary in each axis direction.
    // For a positive step, the far face of the current cell is at index+1;
    // for a negative step, the far face is at the current index.
    int nx_x = ix + (step_x > 0 ? 1 : 0);
    int nx_y = iy + (step_y > 0 ? 1 : 0);
    int nx_z = iz + (step_z > 0 ? 1 : 0);

    // Parametric t at which the ray first crosses each axis's next boundary.
    float t_max_x = safe_div(boundary_in_world_x(nx_x) - camera_pos.x, dir_normalized.x);
    float t_max_y = safe_div(boundary_in_world_y(nx_y) - camera_pos.y, dir_normalized.y);
    float t_max_z = safe_div(boundary_in_world_z(nx_z) - camera_pos.z, dir_normalized.z);

    // Parametric distance the ray must travel to cross a full voxel in each axis.
    // Used to increment t_max_* after each step.
    float t_delta_x = safe_div(voxel_size, std::fabs(dir_normalized.x));
    float t_delta_y = safe_div(voxel_size, std::fabs(dir_normalized.y));
    float t_delta_z = safe_div(voxel_size, std::fabs(dir_normalized.z));

    float t_current = t_min;  // Parametric position at the leading edge of traversal.
    int   step_count = 0;

    // Main DDA loop: advance one voxel at a time until we exit the grid AABB.
    while(t_current <= t_max) {
        // Record the current voxel before advancing.
        RayStep rs;
        rs.ix         = ix;
        rs.iy         = iy;
        rs.iz         = iz;
        rs.step_count = step_count;
        rs.distance   = t_current;
        steps.push_back(rs);

        // Advance along the axis whose next boundary is closest.
        // This ensures we visit every voxel the ray intersects with no
        // skips or double-counts.
        if(t_max_x < t_max_y && t_max_x < t_max_z) {
            // X boundary is nearest: step in X.
            ix        += step_x;
            t_current  = t_max_x;
            t_max_x   += t_delta_x;
        } else if(t_max_y < t_max_z) {
            // Y boundary is nearest: step in Y.
            iy        += step_y;
            t_current  = t_max_y;
            t_max_y   += t_delta_y;
        } else {
            // Z boundary is nearest (or tied): step in Z.
            iz        += step_z;
            t_current  = t_max_z;
            t_max_z   += t_delta_z;
        }

        step_count++;

        // Exit loop if we've stepped out of the valid grid volume.
        if(ix < 0 || ix >= N || iy < 0 || iy >= N || iz < 0 || iz >= N) {
            break;
        }
    }

    return steps;
}


// ============================================================
// Section 7: Main Pipeline
// ============================================================

/**
 * @brief Program entry point. Orchestrates the full motion-to-voxel pipeline.
 *
 * ### Pipeline Summary
 * 1. Parse command-line arguments and load the JSON metadata file.
 * 2. Group frames by camera and sort each group by frame index.
 * 3. Allocate a sparse voxel grid (hash map of flat index → float value).
 * 4. For each camera:
 *    a. Load consecutive frame pairs and compute a motion mask.
 *    b. For each changed pixel, build a camera-to-world ray.
 *    c. Run DDA traversal through the voxel grid.
 *    d. Accumulate the motion-difference value into each traversed voxel.
 * 5. Write the voxel grid to a binary file.
 *
 * ### Ray Construction (step 4b)
 * A pinhole camera model is used. The focal length in pixels is derived from
 * the horizontal FOV:
 * @code
 *   focal_length_px = (image_width / 2) / tan(fov_horizontal / 2)
 * @endcode
 * Each pixel (u, v) maps to a camera-space direction:
 * @code
 *   ray_cam = normalize([ u - cx,  -(v - cy),  -focal_length_px ])
 * @endcode
 * where (cx, cy) = (width/2, height/2) is the principal point, and the
 * negative Y flips from image-row-down to camera-Y-up convention.
 * The negative Z places the image plane in front of the camera.
 *
 * The camera-space ray is then rotated to world space:
 * @code
 *   ray_world = normalize(camera_rotation_matrix * ray_cam)
 * @endcode
 *
 * ### Voxel Accumulation (step 4d)
 * Each traversed voxel receives the raw motion-difference value of the
 * triggering pixel. After all frames and cameras are processed, voxels in
 * regions consistently seen as "moving" by many cameras accumulate large
 * values, forming a probabilistic occupancy map.
 *
 * @note The distance-based attenuation factor `alpha` is computed but
 * currently not applied (the accumulation uses a fixed weight of 1.0).
 * Enabling it would weight near voxels more than far ones, which may or
 * may not improve reconstruction quality depending on the scene scale.
 *
 * @param argc Argument count. Must be 4.
 * @param argv Argument vector: [program, metadata.json, image_folder, output.bin]
 * @return 0 on success, 1 on error (missing args, empty metadata, I/O failure).
 */
int main(int argc, char** argv) {
    if(argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <metadata.json> <image_folder> <output_voxel_bin>\n";
        return 1;
    }

    std::string metadata_path = argv[1];
    std::string images_folder = argv[2];
    std::string output_bin    = argv[3];

    // -------------------------------------------------------
    // Step 7.1: Load and organise frame metadata
    // -------------------------------------------------------
    std::vector<FrameInfo> frames = load_metadata(metadata_path);
    if(frames.empty()) {
        std::cerr << "No frames loaded.\n";
        return 1;
    }

    // Group frames by camera so we can process consecutive pairs per camera.
    std::map<int, std::vector<FrameInfo>> frames_by_cam;
    for(const auto &f : frames) {
        frames_by_cam[f.camera_index].push_back(f);
    }

    // Sort each camera's frames chronologically by frame_index.
    // This ensures we always diff frame[i] with frame[i+1], not in random order.
    for(auto &kv : frames_by_cam) {
        auto &v = kv.second;
        std::sort(v.begin(), v.end(), [](auto &a, auto &b) {
            return a.frame_index < b.frame_index;
        });
    }

    // -------------------------------------------------------
    // Step 7.2: Load profile and resolve all parameters
    // -------------------------------------------------------
    Profile prof = load_profile(metadata_path);

    // Auto-centre: compute centroid of all camera positions in the metadata.
    if(prof.center_auto) {
        double sx = 0, sy = 0, sz = 0;
        for(const auto &f : frames) {
            sx += f.camera_position.x;
            sy += f.camera_position.y;
            sz += f.camera_position.z;
        }
        double n = static_cast<double>(frames.size());
        prof.grid_center = {
            static_cast<float>(sx / n),
            static_cast<float>(sy / n),
            static_cast<float>(sz / n)
        };
        std::cout << "[Profile] Auto-centred grid at ("
                  << prof.grid_center.x << ", "
                  << prof.grid_center.y << ", "
                  << prof.grid_center.z << ")\n";
    }

    const int   N          = prof.resolution;
    const float voxel_size = prof.voxel_size;
    const Vec3  grid_center = prof.grid_center;

    std::cout << "[Profile] Grid: " << N << "^3, voxel_size=" << voxel_size
              << ", center=(" << grid_center.x << "," << grid_center.y
              << "," << grid_center.z << ")\n";

    // -------------------------------------------------------
    // Step 7.3: Allocate sparse voxel grid
    // -------------------------------------------------------
    // Only voxels touched by at least one ray are stored.
    std::unordered_map<int, float> voxel_grid;
    std::unordered_map<int, std::unordered_set<int>> camera_hits;

    // -------------------------------------------------------
    // Step 7.4: Per-camera motion detection and ray accumulation
    // -------------------------------------------------------
    for(auto &kv : frames_by_cam) {
        int  cam_id     = kv.first;
        auto &cam_frames = kv.second;

        // Resolve per-camera overrides, falling back to global profile values.
        float cam_threshold = prof.pixel_diff_threshold;
        int   cam_stride    = prof.frame_stride;
        auto ov_it = prof.camera_overrides.find(cam_id);
        if(ov_it != prof.camera_overrides.end()) {
            if(ov_it->second.pixel_diff_threshold >= 0.f)
                cam_threshold = ov_it->second.pixel_diff_threshold;
            if(ov_it->second.frame_stride > 0)
                cam_stride = ov_it->second.frame_stride;
        }

        // Motion detection requires at least two frames separated by cam_stride.
        if((int)cam_frames.size() < cam_stride + 1) {
            continue;
        }

        // Build a strided index list: process pairs (0, stride), (stride, 2*stride), ...
        // Each "prev" frame is loaded once and reused as the anchor for the next pair.
        std::vector<size_t> pair_starts;
        for(size_t i = 0; i + cam_stride < cam_frames.size(); i += cam_stride)
            pair_starts.push_back(i);

        for(size_t pi = 0; pi < pair_starts.size(); pi++) {
            size_t prev_idx = pair_starts[pi];
            size_t curr_idx = prev_idx + cam_stride;

            std::string prev_path = images_folder + "/" + cam_frames[prev_idx].image_file;
            std::string curr_path = images_folder + "/" + cam_frames[curr_idx].image_file;

            ImageGray prev_img, curr_img;
            if(!load_image_gray(prev_path, prev_img)) {
                std::cerr << "Skipping pair: failed to load " << prev_path << "\n";
                continue;
            }
            if(!load_image_gray(curr_path, curr_img)) {
                std::cerr << "Skipping pair: failed to load " << curr_path << "\n";
                continue;
            }

            // Compute motion mask between the strided frame pair.
            MotionMask mm = detect_motion(prev_img, curr_img, cam_threshold);

            // Build the camera model for the current frame.
            const FrameInfo &curr_info = cam_frames[curr_idx];
            Vec3 cam_pos  = curr_info.camera_position;
            Mat3 cam_rot  = camera_matrix_from_generator_convention(
                                curr_info.yaw, curr_info.pitch, curr_info.roll);
            float fov_rad   = deg2rad(curr_info.fov_degrees);

            // Derive focal length from the horizontal FOV and image width.
            // This places the image plane at a distance focal_len pixels from
            // the camera centre, matching the standard pinhole model.
            float focal_len = (mm.width * 0.5f) / std::tan(fov_rad * 0.5f);

            // Iterate over every pixel in the motion mask.
            for(int v = 0; v < mm.height; v++) {
                for(int u = 0; u < mm.width; u++) {
                    if(!mm.changed[v * mm.width + u]) {
                        continue; // Pixel is static — skip.
                    }

                    // Motion signal strength: use raw absolute difference as proxy
                    // for how "confidently" something was moving at this pixel.
                    float pix_val = mm.diff[v * mm.width + u];
                    if(pix_val < 1e-3f) {
                        continue; // Negligible difference — skip.
                    }

                    // Build the camera-space ray direction for pixel (u, v).
                    //   - Origin at image centre (cx, cy) = (width/2, height/2).
                    //   - Y is flipped: image rows increase downward, but camera Y is up.
                    //   - Z is negative because the image plane is in front of the camera
                    //     and we use a right-handed camera-space convention.
                    float rx = float(u) - 0.5f * mm.width;
                    float ry = -(float(v) - 0.5f * mm.height);  // flip Y for screen->camera
                    float rz = -focal_len;

                    Vec3 ray_cam = normalize({rx, ry, rz});

                    // Rotate camera-space direction into world space using the camera's
                    // rotation matrix (built from yaw/pitch/roll Euler angles).
                    Vec3 ray_world = normalize(mat3_mul_vec3(cam_rot, ray_cam));

                    // Traverse the voxel grid along ray_world using 3D DDA.
                    std::vector<RayStep> ray_steps = cast_ray_into_grid(
                        cam_pos, ray_world, N, voxel_size, grid_center);

                    // Accumulate motion signal into each traversed voxel.
                    // max_steps=0 means unlimited; otherwise truncate the ray.
                    // TODO: The distance-attenuation weight (1/(1+alpha*dist)) is
                    // computed here but val is fixed at pix_val * 1.0 for now.
                    // Enabling attenuation would down-weight distant voxels, which
                    // could improve localisation but needs tuning to avoid biasing
                    // reconstruction toward the cameras.
                    int step_limit = (prof.max_steps > 0)
                                     ? std::min((int)ray_steps.size(), prof.max_steps)
                                     : (int)ray_steps.size();
                    for(int si = 0; si < step_limit; si++) {
                        const auto &rs = ray_steps[si];
                        float dist        = rs.distance;
                        float attenuation = 1.f / (1.f + prof.attenuation_coefficient * dist); // currently unused
                        float val         = pix_val * 1.f;              // TODO: * attenuation

                        // Flat index into the voxel grid: [ix][iy][iz].
                        int idx = rs.ix * N * N + rs.iy * N + rs.iz;
                        voxel_grid[idx] += val;
                        // Record this camera as a contributor to the voxel.
                        camera_hits[idx].insert(cam_id);
                    }
                }
            }

        }
    }

    // -------------------------------------------------------
    // Step 7.5: Write sparse voxel grid to binary file
    // -------------------------------------------------------
    /**
     * Sparse binary output format:
     *   Bytes  0– 3:  int   N           (grid resolution, voxels per axis)
     *   Bytes  4– 7:  float voxel_size  (side length of one voxel, world units)
     *   Bytes  8–11:  int   count       (number of non-zero voxel entries)
     *   Bytes 12–...: entry[count]      (each entry = int flat_index + float value = 8 bytes)
     *
     * Total file size = 12 + count * 8 bytes, where count << N^3 for typical scenes.
     * Example: 100k active voxels → ~800 KB vs ~500 MB for the dense equivalent.
     *
     * Flat index encoding: flat_index = ix*N*N + iy*N + iz
     * Readers can recover (ix, iy, iz) via:
     *   ix = flat_index / (N*N)
     *   iy = (flat_index / N) % N
     *   iz = flat_index % N
     *
     * The header values (N, voxel_size) allow downstream readers to reconstruct
     * world-space voxel positions without needing the original metadata.
     */
    {
        std::ofstream ofs(output_bin, std::ios::binary);
        if(!ofs) {
            std::cerr << "Cannot open output file: " << output_bin << "\n";
            return 1;
        }

        // Write header: grid resolution and voxel size.
        ofs.write(reinterpret_cast<const char*>(&N),          sizeof(int));
        ofs.write(reinterpret_cast<const char*>(&voxel_size), sizeof(float));

        // Write entry count so readers can pre-allocate.
        int count = static_cast<int>(voxel_grid.size());
        ofs.write(reinterpret_cast<const char*>(&count), sizeof(int));

        // Write each (flat_index, value) pair.
        for(const auto &kv : voxel_grid) {
            int   idx = kv.first;
            float val = kv.second;
            ofs.write(reinterpret_cast<const char*>(&idx), sizeof(int));
            ofs.write(reinterpret_cast<const char*>(&val), sizeof(float));
        }

        ofs.close();
        std::cout << "Saved sparse voxel grid to " << output_bin
                  << " (" << count << " active voxels, "
                  << (12 + count * 8) / 1024 << " KB)\n";
    }

    return 0;
}