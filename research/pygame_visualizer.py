"""
Interactive Pygame Trajectory Visualizer for 2-Stage Kinematic Acceleration System
Stage 1: Rest vs Moving Classifier (MLP Zero-Velocity Detector)
Stage 2: Acceleration (Δv) Transformer + Kinematic Velocity Integration + 3D Heading Tracking
"""

import os
import sys
import math
import json
import numpy as np
import pandas as pd
import torch
import pygame

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from dataset import parse_and_clean_imu_data, download_dataset, Normalizer
from models import RestMovingClassifierMLP, IMUTransformerTLIO

RESEARCH_DIR = os.path.dirname(os.path.abspath(__file__))


def load_models_and_scalers():
    scaler_path = os.path.join(RESEARCH_DIR, "scaler_params.json")
    with open(scaler_path, "r") as f:
        scaler_dict = json.load(f)

    feat_norm = Normalizer()
    feat_norm.mean = np.array(scaler_dict['features']['mean'], dtype=np.float32)
    feat_norm.std = np.array(scaler_dict['features']['std'], dtype=np.float32)

    target_norm = Normalizer()
    target_norm.mean = np.array(scaler_dict['targets']['mean'], dtype=np.float32)
    target_norm.std = np.array(scaler_dict['targets']['std'], dtype=np.float32)

    cls_model = RestMovingClassifierMLP(input_dim=6, window_size=10, hidden_dim=64)
    cls_pt = os.path.join(RESEARCH_DIR, "motion_classifier.pt")
    cls_model.load_state_dict(torch.load(cls_pt, map_location="cpu"))
    cls_model.eval()

    trans_model = IMUTransformerTLIO(
        input_dim=6,
        window_size=10,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.0,
        output_dim=2
    )
    trans_pt = os.path.join(RESEARCH_DIR, "tlio_transformer.pt")
    trans_model.load_state_dict(torch.load(trans_pt, map_location="cpu"))
    trans_model.eval()

    return cls_model, trans_model, feat_norm, target_norm


def precompute_trajectory(dataset_key="S-S1", max_seconds=180):
    cls_model, trans_model, feat_norm, target_norm = load_models_and_scalers()
    csv_path = download_dataset(dataset_key)
    raw_df = pd.read_csv(csv_path, encoding='latin1')
    raw_df.columns = [c.strip() for c in raw_df.columns]

    for c in raw_df.columns:
        if 'gps orientation' in c.lower() or ('orientation' in c.lower() and 'gps' in c.lower()): gps_h_col = c
        if 'orientation (yaw)' in c.lower(): phone_yaw_col = c
        if 'gps speed' in c.lower(): gps_sp_col = c
        if 'accel' in c.lower() and 'x' in c.lower(): ax_col = c
        if 'accel' in c.lower() and 'y' in c.lower(): ay_col = c
        if 'accel' in c.lower() and 'z' in c.lower(): az_col = c
        if 'gyro' in c.lower() and 'roll' in c.lower(): gx_col = c
        if 'gyro' in c.lower() and 'pitch' in c.lower(): gy_col = c
        if 'gyro' in c.lower() and 'yaw' in c.lower(): gz_col = c

    window_size = 10
    total_samples = min(len(raw_df), max_seconds * 10)
    sample_df = raw_df.iloc[:total_samples].reset_index(drop=True)

    features = sample_df[[ax_col, ay_col, az_col, gx_col, gy_col, gz_col]].values.astype(np.float32)
    norm_features = feat_norm.transform(features)

    speeds_mps = (sample_df[gps_sp_col].values / 3.6).astype(np.float32)
    gps_headings = sample_df[gps_h_col].values.astype(np.float32)
    phone_yaws = sample_df[phone_yaw_col].values.astype(np.float32)

    gt_x_all = [0.0]
    gt_y_all = [0.0]
    for i in range(len(sample_df) - 1):
        step = speeds_mps[i] * 0.1
        h_rad = np.radians(gps_headings[i])
        gt_x_all.append(gt_x_all[-1] + step * np.sin(h_rad))
        gt_y_all.append(gt_y_all[-1] + step * np.cos(h_rad))

    start_bearing_deg = float(gps_headings[0])
    initial_phone_yaw = float(phone_yaws[0])

    n_seconds = total_samples // window_size
    pred_steps = [[0.0, 0.0]]
    gt_steps = [[0.0, 0.0]]
    speeds_gt = [0.0]
    speeds_pred = [0.0]
    motion_states = ["REST"]

    cur_pred_x = 0.0
    cur_pred_y = 0.0
    cur_v_mps = float(speeds_mps[0])

    print(f"[Pygame] Precomputing {n_seconds} seconds using 2-Stage Kinematic Acceleration System...")
    with torch.no_grad():
        for s in range(n_seconds):
            start = s * window_size
            end = start + window_size
            w_x = norm_features[start:end]
            tensor_x = torch.tensor(w_x, dtype=torch.float32).unsqueeze(0)

            # STAGE 1: Rest vs Moving Classifier
            logits = cls_model(tensor_x)
            is_moving = int(torch.argmax(logits, dim=1).item())

            # STAGE 2: Acceleration Prediction & Kinematic Velocity Integration
            v_prev = cur_v_mps
            if is_moving == 1:
                pred_norm = trans_model(tensor_x).numpy()[0]
                disp = target_norm.inverse_transform(pred_norm)
                a_fwd = float(disp[1])
                cur_v_mps = max(0.0, cur_v_mps + a_fwd)
                m_state = "MOVING"
            else:
                cur_v_mps = 0.0
                m_state = "REST"

            motion_states.append(m_state)

            fwd_disp = ((v_prev + cur_v_mps) / 2.0) * 1.0

            # 3D Vehicle Heading Tracking
            cur_phone_yaw = float(phone_yaws[end - 1])
            delta_yaw = (cur_phone_yaw - initial_phone_yaw + 180) % 360 - 180
            cur_heading_deg = (start_bearing_deg - delta_yaw) % 360
            cur_heading_rad = np.radians(cur_heading_deg)

            dx_w = fwd_disp * np.sin(cur_heading_rad)
            dy_w = fwd_disp * np.cos(cur_heading_rad)

            cur_pred_x += dx_w
            cur_pred_y += dy_w

            pred_steps.append([cur_pred_x, cur_pred_y])
            gt_steps.append([gt_x_all[end - 1], gt_y_all[end - 1]])

            speeds_pred.append(cur_v_mps * 3.6)
            speeds_gt.append(float(np.mean(speeds_mps[start:end])) * 3.6)

    return {
        'n_seconds': n_seconds,
        'pred_steps': np.array(pred_steps, dtype=np.float32),
        'gt_steps': np.array(gt_steps, dtype=np.float32),
        'speeds_pred': np.array(speeds_pred, dtype=np.float32),
        'speeds_gt': np.array(speeds_gt, dtype=np.float32),
        'motion_states': motion_states
    }


def run_pygame_visualizer(dataset_key="S-S1", max_seconds=180):
    pygame.init()
    pygame.display.set_caption("IMU-Sync // 2-Stage Pygame Visualizer (Kinematic Acceleration + 3D Orientation)")

    WIDTH, HEIGHT = 1280, 720
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    clock = pygame.time.Clock()

    data = precompute_trajectory(dataset_key, max_seconds)
    n_seconds = data['n_seconds']
    pred_steps = data['pred_steps']
    gt_steps = data['gt_steps']
    speeds_pred = data['speeds_pred']
    speeds_gt = data['speeds_gt']
    motion_states = data['motion_states']

    BG_COLOR = (13, 17, 23)
    GRID_COLOR = (30, 36, 46)
    AXIS_COLOR = (45, 55, 72)
    CYAN = (0, 240, 255)
    GREEN = (63, 185, 80)
    AMBER = (255, 184, 0)
    RED = (248, 81, 73)
    WHITE = (240, 246, 252)
    MUTED = (139, 148, 158)
    CARD_BG = (22, 27, 34, 230)

    font_main = pygame.font.SysFont("Consolas", 14)
    font_title = pygame.font.SysFont("Consolas", 16, bold=True)

    zoom = 6.0
    cam_x = 0.0
    cam_y = 0.0
    is_dragging = False
    drag_start = (0, 0)
    drag_cam_start = (0.0, 0.0)
    auto_follow = True

    current_time_sec = 0.0
    playback_speed = 1.0
    is_playing = True

    def world_to_screen(wx, wy, cur_w, cur_h):
        sx = (cur_w / 2) + (wx - cam_x) * zoom
        sy = (cur_h / 2) - (wy - cam_y) * zoom
        return int(sx), int(sy)

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        cur_w, cur_h = screen.get_size()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    is_playing = not is_playing
                elif event.key == pygame.K_r:
                    current_time_sec = 0.0
                elif event.key == pygame.K_c:
                    auto_follow = True
                elif event.key == pygame.K_1:
                    playback_speed = 1.0
                elif event.key == pygame.K_2:
                    playback_speed = 2.0
                elif event.key == pygame.K_3:
                    playback_speed = 5.0
                elif event.key == pygame.K_4:
                    playback_speed = 10.0
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    is_dragging = True
                    drag_start = event.pos
                    drag_cam_start = (cam_x, cam_y)
                    auto_follow = False
                elif event.button == 4:
                    zoom = min(zoom * 1.15, 50.0)
                elif event.button == 5:
                    zoom = max(zoom / 1.15, 0.5)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    is_dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if is_dragging:
                    dx = event.pos[0] - drag_start[0]
                    dy = event.pos[1] - drag_start[1]
                    cam_x = drag_cam_start[0] - dx / zoom
                    cam_y = drag_cam_start[1] + dy / zoom

        if is_playing:
            current_time_sec += dt * playback_speed
            if current_time_sec >= n_seconds:
                current_time_sec = n_seconds

        step_idx = int(current_time_sec)
        alpha = current_time_sec - step_idx

        if step_idx < n_seconds:
            pred_p0 = pred_steps[step_idx]
            pred_p1 = pred_steps[step_idx + 1]
            cur_pred = (1 - alpha) * pred_p0 + alpha * pred_p1

            gt_p0 = gt_steps[step_idx]
            gt_p1 = gt_steps[step_idx + 1]
            cur_gt = (1 - alpha) * gt_p0 + alpha * gt_p1

            cur_spd_pred = (1 - alpha) * speeds_pred[step_idx] + alpha * speeds_pred[step_idx + 1]
            cur_spd_gt = (1 - alpha) * speeds_gt[step_idx] + alpha * speeds_gt[step_idx + 1]
            cur_state = motion_states[step_idx]
        else:
            cur_pred = pred_steps[-1]
            cur_gt = gt_steps[-1]
            cur_spd_pred = speeds_pred[-1]
            cur_spd_gt = speeds_gt[-1]
            cur_state = motion_states[-1]

        if auto_follow:
            cam_x += (cur_pred[0] - cam_x) * 0.1
            cam_y += (cur_pred[1] - cam_y) * 0.1

        screen.fill(BG_COLOR)

        grid_meter_step = 10.0 if zoom > 3.0 else 50.0
        grid_px = grid_meter_step * zoom
        offset_x = (cur_w / 2 - cam_x * zoom) % grid_px
        offset_y = (cur_h / 2 + cam_y * zoom) % grid_px

        for x in range(int(offset_x), cur_w, int(grid_px)):
            pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, cur_h), 1)
        for y in range(int(offset_y), cur_h, int(grid_px)):
            pygame.draw.line(screen, GRID_COLOR, (0, y), (cur_w, y), 1)

        ox, oy = world_to_screen(0, 0, cur_w, cur_h)
        pygame.draw.line(screen, AXIS_COLOR, (ox, 0), (ox, cur_h), 2)
        pygame.draw.line(screen, AXIS_COLOR, (0, oy), (cur_w, oy), 2)
        pygame.draw.circle(screen, WHITE, (ox, oy), 4)

        if len(gt_steps) > 1:
            gt_pts_full = [world_to_screen(p[0], p[1], cur_w, cur_h) for p in gt_steps]
            pred_pts_full = [world_to_screen(p[0], p[1], cur_w, cur_h) for p in pred_steps]
            pygame.draw.lines(screen, (35, 75, 45), False, gt_pts_full, 1)
            pygame.draw.lines(screen, (0, 70, 85), False, pred_pts_full, 1)

        active_gt_pts = [world_to_screen(p[0], p[1], cur_w, cur_h) for p in gt_steps[:step_idx + 1]]
        active_gt_pts.append(world_to_screen(cur_gt[0], cur_gt[1], cur_w, cur_h))
        if len(active_gt_pts) > 1:
            pygame.draw.lines(screen, GREEN, False, active_gt_pts, 3)

        active_pred_pts = [world_to_screen(p[0], p[1], cur_w, cur_h) for p in pred_steps[:step_idx + 1]]
        active_pred_pts.append(world_to_screen(cur_pred[0], cur_pred[1], cur_w, cur_h))
        if len(active_pred_pts) > 1:
            pygame.draw.lines(screen, CYAN, False, active_pred_pts, 3)

        gt_head = world_to_screen(cur_gt[0], cur_gt[1], cur_w, cur_h)
        pygame.draw.circle(screen, GREEN, gt_head, 6)
        pygame.draw.circle(screen, WHITE, gt_head, 2)

        pred_head = world_to_screen(cur_pred[0], cur_pred[1], cur_w, cur_h)
        pygame.draw.circle(screen, CYAN, pred_head, 8, 2)
        pygame.draw.circle(screen, CYAN, pred_head, 3)

        px, py = cur_pred[0], cur_pred[1]
        dist = math.hypot(px, py)
        dir_x, dir_y = (px / dist, -py / dist) if dist > 0.001 else (0.0, -1.0)

        arrow_len = 28
        tip_x = pred_head[0] + dir_x * arrow_len
        tip_y = pred_head[1] + dir_y * arrow_len
        pygame.draw.line(screen, CYAN, pred_head, (tip_x, tip_y), 2)

        angle = math.atan2(dir_y, dir_x)
        w_len = 8
        w_ang = math.pi / 6
        p1 = (tip_x - w_len * math.cos(angle - w_ang), tip_y - w_len * math.sin(angle - w_ang))
        p2 = (tip_x - w_len * math.cos(angle + w_ang), tip_y - w_len * math.sin(angle + w_ang))
        pygame.draw.polygon(screen, AMBER, [(tip_x, tip_y), p1, p2])

        pygame.draw.line(screen, (248, 81, 73, 160), gt_head, pred_head, 1)

        hud_surface = pygame.Surface((400, 240), pygame.SRCALPHA)
        hud_surface.fill(CARD_BG)
        pygame.draw.rect(hud_surface, (48, 54, 61), (0, 0, 400, 240), 1, border_radius=8)

        ate = math.hypot(cur_pred[0] - cur_gt[0], cur_pred[1] - cur_gt[1])

        t_title = font_title.render("IMU-SYNC // KINEMATIC ACCELERATION", True, CYAN)
        hud_surface.blit(t_title, (12, 10))

        lines = [
            (f"TIME: {int(current_time_sec):03d}s / {n_seconds:03d}s  |  SPEED: {playback_speed:.0f}x", WHITE),
            (f"MOTION STATE: [{cur_state}] (Zero-Velocity Gated)", GREEN if cur_state == 'MOVING' else AMBER),
            (f"GROUND TRUTH: ({cur_gt[0]:.2f}m, {cur_gt[1]:.2f}m) @ {cur_spd_gt:.1f} km/h", GREEN),
            (f"KINEMATIC AI: ({cur_pred[0]:.2f}m, {cur_pred[1]:.2f}m) @ {cur_spd_pred:.1f} km/h", CYAN),
            (f"DRIFT ATE:    {ate:.2f} meters", RED if ate > 10 else AMBER),
            (f"CAMERA:       Zoom {zoom:.1f}px/m  |  Auto-Follow: {'ON' if auto_follow else 'OFF'}", MUTED),
            (f"CONTROLS:     [SPACE] Play/Pause  [R] Restart  [C] Recenter", MUTED),
            (f"              [1-4] Speed (1x, 2x, 5x, 10x)  [Drag] Pan", MUTED)
        ]

        y_off = 35
        for text, color in lines:
            rendered = font_main.render(text, True, color)
            hud_surface.blit(rendered, (12, y_off))
            y_off += 21

        screen.blit(hud_surface, (15, 15))

        leg_surface = pygame.Surface((280, 70), pygame.SRCALPHA)
        leg_surface.fill(CARD_BG)
        pygame.draw.rect(leg_surface, (48, 54, 61), (0, 0, 280, 70), 1, border_radius=6)

        pygame.draw.line(leg_surface, GREEN, (15, 22), (45, 22), 3)
        pygame.draw.circle(leg_surface, GREEN, (52, 22), 4)
        t_gt = font_main.render("Ground Truth Route", True, WHITE)
        leg_surface.blit(t_gt, (65, 14))

        pygame.draw.line(leg_surface, CYAN, (15, 48), (45, 48), 3)
        pygame.draw.circle(leg_surface, CYAN, (52, 48), 4)
        t_pred = font_main.render("Kinematic AI Path", True, WHITE)
        leg_surface.blit(t_pred, (65, 40))

        screen.blit(leg_surface, (15, cur_h - 85))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="2-Stage Pygame Kinematic Acceleration Visualizer")
    parser.add_argument("--dataset", type=str, default="S-S1", help="Dataset key (S-S1, S-S2, S-M)")
    parser.add_argument("--seconds", type=int, default=180, help="Number of seconds of driving to evaluate")
    args = parser.parse_args()

    run_pygame_visualizer(dataset_key=args.dataset, max_seconds=args.seconds)
