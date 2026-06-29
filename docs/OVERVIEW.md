# Diffusion Policy — 시스템 구조 & 학습 Overview

이 문서는 코드를 처음 보고 학습을 돌릴 때 "어느 단계에서 어떤 스크립트를 봐야 하는지"를 정리한 지도다.

## 1. 한눈에 보는 구조

```
train.py  ──hydra──▶  Workspace.run()  ─────────────────┐
 (엔트리)              (학습 루프 전체 오케스트레이션)        │
                                                         │ 매 epoch:
   ┌───────────── 아래 4개를 cfg로 instantiate ──────────┤
   │                                                     ▼
 Policy        Dataset          EnvRunner          Checkpoint/Logger
 (모델)        (데이터+정규화)   (시뮬 rollout 평가)  (TopK ckpt, wandb, json)
```

핵심 설계: **모든 객체는 코드가 아니라 YAML config의 `_target_` 로 생성된다 (Hydra).**
즉 "무엇을 학습하는가"는 코드 수정이 아니라 config 선택/오버라이드로 결정된다.

## 2. 디렉토리 역할

| 경로 | 역할 |
|---|---|
| `train.py` | 엔트리포인트. config 읽어 workspace 클래스 instantiate 후 `.run()` |
| `eval.py` | 학습된 체크포인트 로드 → EnvRunner로 평가만 수행 |
| `diffusion_policy/config/` | **모든 실험 설정.** `train_*.yaml` = 실험 전체, `task/*.yaml` = 데이터셋+환경 |
| `diffusion_policy/workspace/` | 학습 루프 본체 (epoch 반복, loss, eval, checkpoint) |
| `diffusion_policy/policy/` | 모델 (forward = `compute_loss`, inference = `predict_action`) |
| `diffusion_policy/model/` | 네트워크 빌딩블록 (UNet1D, transformer, vision encoder, EMA 등) |
| `diffusion_policy/dataset/` | 데모 데이터 로딩 + `get_normalizer()` / `get_validation_dataset()` |
| `diffusion_policy/env_runner/` | 시뮬 환경에서 policy rollout → success rate 측정 |
| `diffusion_policy/env/` | 시뮬 환경 정의 (pusht, robomimic 등) |
| `diffusion_policy/common/` | 유틸 (replay_buffer, sampler, normalizer, json_logger, checkpoint_util) |
| `diffusion_policy/real_world/` | 실제 로봇 (UR5, RealSense, SpaceMouse) 관련 |

## 3. 학습 단계별 — 어떤 스크립트를 보면 되는가

### 단계 0: 실험 정의 (config)
- **`config/train_*_workspace.yaml`** ← 먼저 여기. horizon, obs/action step, optimizer, epoch 수, EMA, checkpoint 정책이 다 들어있다.
- `defaults: - task: <name>` 으로 **`config/task/*.yaml`** 을 끌어온다. task 파일에는:
  - `shape_meta`: obs 키별 shape/type(rgb/low_dim), action shape
  - `dataset:` (`_target_` = 어떤 Dataset 클래스 + 경로)
  - `env_runner:` (`_target_` = 어떤 평가 러너 + n_test, max_steps 등)
- 예: `train_diffusion_unet_hybrid_workspace.yaml` + `task/lift_image_abs.yaml`

### 단계 1: 진입 & 객체 생성
- **`train.py`**: `cfg._target_` → workspace 클래스 로드, `cfg.policy` 등은 아직 미생성.
- **`workspace/train_*_workspace.py` 의 `__init__`**: seed 고정, `cfg.policy` instantiate(=모델), EMA 복제, optimizer 생성.

### 단계 2: 데이터 준비
- workspace `run()` 안에서 `cfg.task.dataset` instantiate → **`dataset/*.py`**.
- 여기서 `get_normalizer()` 호출 → policy에 주입. 정규화 로직 궁금하면 `common/normalize_util.py`, `model/common/normalizer.py`.
- 시퀀스 샘플링(horizon, pad_before/after)은 **`common/sampler.py` + `common/replay_buffer.py`**.

### 단계 3: 학습 루프 (가장 중요)
**`workspace/train_*_workspace.py` 의 `run()`** — 전부 여기 있다:
- 매 batch: `raw_loss = self.model.compute_loss(batch)` → backward → optimizer.step → `ema.step`.
- loss 내부(노이즈 추가/예측)는 **`policy/diffusion_unet_*_policy.py`** 의 `compute_loss`.
- 실제 네트워크는 **`model/diffusion/conditional_unet1d.py`** 또는 `transformer_for_diffusion.py`.
- 이미지 인코더는 **`model/vision/multi_image_obs_encoder.py`**.

### 단계 4: 평가 (rollout)
- `rollout_every`(기본 50 epoch) 마다 `env_runner.run(policy)`.
- **`env_runner/*_runner.py`**: 환경 여러 개 병렬 실행, `policy.predict_action(obs)` 반복, success rate를 `test/mean_score`로 반환.
- inference 디테일(역확산 sampling)은 policy의 `predict_action` / `conditional_sample`.

### 단계 5: 로깅 & 체크포인트
- wandb + **`common/json_logger.py`** (`logs.json.txt`).
- **`common/checkpoint_util.py` 의 `TopKCheckpointManager`**: `test_mean_score` 기준 상위 k개 저장 + `latest.ckpt`.
- 저장/복원 메커니즘은 **`workspace/base_workspace.py`** (`save_checkpoint`/`load_payload`). cfg까지 통째로 ckpt에 저장 → eval 때 코드 없이 복원 가능.

### 단계 6: 학습된 모델 평가
- **`eval.py`**: ckpt 로드 → `cfg`로 workspace 재생성 → `ema_model` 꺼내 EnvRunner 실행 → `eval_log.json`.

## 4. 실행 방법

```bash
# 학습 (config-name으로 실험 선택, key=value로 오버라이드)
python train.py --config-name=train_diffusion_unet_hybrid_workspace \
    training.seed=42 training.device=cuda:0

# README의 다운로드 config로 학습 (외부 config-dir)
python train.py --config-dir=. --config-name=image_pusht_diffusion_policy_cnn.yaml \
    training.seed=42 training.device=cuda:0

# 디버그(빠른 점검: 2 epoch, 3 step) — 코드 안 고치고 플래그로
python train.py --config-name=... training.debug=True

# 평가
python eval.py -c <ckpt>.ckpt -o data/eval_out
```

출력: `data/outputs/yyyy.mm.dd/hh.mm.ss_<name>_<task>/` 안에 `checkpoints/`, `logs.json.txt`, hydra config.

## 5. 코드 읽는 추천 순서
1. `train.py` (5분) → 2. 돌릴 `config/train_*.yaml` 1개 + 그 `task/*.yaml` →
3. 해당 `workspace/train_*.py` 의 `run()` (학습 루프) →
4. 해당 `policy/*.py` 의 `compute_loss` / `predict_action` →
5. 필요 시 `model/diffusion/`, `dataset/`, `env_runner/` 로 깊게.

> 변형(transformer vs unet, lowdim vs image vs hybrid)은 workspace/policy/config 파일 이름이 1:1로 대응한다. 파일명 규칙만 알면 원하는 조합을 바로 찾을 수 있다.
