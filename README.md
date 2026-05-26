# AeroVoxel

Projects motion of pixels to a voxel

## Execution

```bash
./scripts/build.sh`
# Usage: ray_voxel <metadata.json> <image_folder> <output_voxel_bin>
`./build/ray_voxel metadata.json images build/voxel_grid.bin
```

## Examples

```bash
./examples/synthetic/run.sh
```

## Development

```bash
./scripts/install.sh
. scripts/activate.s
```

## Todo

Figure out PixelationDecensorer.py

## Clean Run
rm build/ray_voxel
./examples/afternoon/clean.sh
./examples/afternoon/run.sh
python src/binToCsv.py examples/afternoon/voxel_grid.bin output.csv

## Calibration Server

version: '3.8'

services:
  # 1. Local DNS Server (Binds to standard network DNS port 53)
  dns:
    image: pihole/pihole:latest
    container_name: lan_dns
    ports:
      - "53:53/udp"
      - "53:53/tcp"
      - "8081:80/tcp" # Access the DNS admin web dashboard here if needed
    environment:
      TZ: 'America/New_York'
      FTLCONF_dns_listeningMode: 'all' # Allow listening on the Docker bridge network
      # Define your custom local DNS mapping rules here directly in code!
      FTLCONF_dns_hosts: |
        192.168.1.50 stream.local
        192.168.1.50 signaling.local
        192.168.1.50 turn.local
    restart: always

  # 2. Frontend Web Server & Reverse Proxy
  webserver:
    image: nginx:alpine
    ports:
      - "443:443"
    volumes:
      - ./frontend:/usr/share/nginx/html
      - ./certs:/etc/nginx/certs
    restart: always
    depends_on:
      - dns

  # 3. Signaling Server
  signaling:
    build: ./signaling-server
    ports:
      - "8443:8443"
    environment:
      - NODE_ENV=production
    restart: always

  # 4. STUN/TURN Media Server
  coturn:
    image: coturn/coturn
    ports:
      - "3478:3478/udp"
      - "3478:3478/tcp"
      - "49152-49170:49152-49170/udp"
    volumes:
      - ./coturn/turnserver.conf:/etc/turnserver.conf
    restart: always