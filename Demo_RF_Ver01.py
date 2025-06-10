import os
import re
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
from scipy.stats import mode
from scipy.ndimage import median_filter
from functools import partial


"""
====================================================
======================= 설정 =======================
====================================================
"""

# wav_paths: 분리 작업을 수행할 오디오 트랙 파일 경로들의 리스트
# 추후 해당 부분은 Demucs와 자동으로 연결되도록 할 예정.
wav_paths = [
    "unwelcomeSchool/vocals.wav",
    "unwelcomeSchool/bass.wav",
    "unwelcomeSchool/drums.wav",
    "unwelcomeSchool/other.wav"
]

# preferred_order: 여러 트랙 중 우선순위를 설정 (0=vocals,1=bass,2=drums,3=other)
# 이 리스트는 인덱스 순서로 우선순위를 정의합니다. 여기선 'other' (3번 트랙)에 제일 높은 우선순위.
preferred_order = [3, 2, 1, 0]
# energy_threshold_ratio: 우선순위 필터를 적용할 때 사용할 에너지 임계 비율
energy_threshold_ratio = 0.05

# 키음이 없는 상태의 채보 데이터
osu_input = "UnwelcomeSchool_Hard.txt"
# base_name: osu 파일 이름에서 확장자를 제거한 기본 이름 (출력 파일 이름 생성에 사용)
base_name = os.path.splitext(os.path.basename(osu_input))[0]

# 트랙 증폭 설정, 몇번 트랙을 몇배 증폭할지
amplify_indices = [3]
amplify_factor = 1.5

# 기본 자르기 ms. BPM기반 연산을 통해 자동 계산 혹은 원곡의 BPM을 알고 있다면 수정.(2안 추천)
default_clip_ms = 333

# 앞 뒤, FadeIn, Out
# 5% 비율로 
fade_ratio = 0.05


"""
====================================================
===================== 파일 로드 =====================
====================================================
"""

# --- 오디오 파일 로드 ---

# waves: 각 트랙의 오디오 데이터를 담을 리스트
waves = []
sr = None  # sr: 오디오 샘플링 레이트(샘플/초)

# wav_paths에 있는 파일을 순회하며 로드
for idx, p in enumerate(wav_paths):
    # librosa.load를 사용하여 오디오 로드 (sr=None으로 원본 샘플링 레이트 유지)
    y, sr_tmp = librosa.load(p, sr=None)
    if sr is None:
        sr = sr_tmp  # 첫 파일의 샘플링 레이트를 sr 변수에 설정
    elif sr != sr_tmp:
        # 만약 새로운 파일의 샘플레이트가 이전과 다르면 에러 발생
        raise ValueError("샘플링 레이트 불일치")
    # amplify_indices에 해당하는 트랙은 amplify_factor 배율로 증폭
    if idx in amplify_indices:
        # 음량 증폭 후 -1.0, 1.0 범위로 클리핑
        y = np.clip(y * amplify_factor, -1.0, 1.0)
    waves.append(y)  # 로드된 오디오 데이터를 waves 리스트에 추가


# --- osu 파일 파싱 ---

# 참고 : osu파일의 [HitObjects]구조
# [HitObjects]
# x,y,time,type,hitSound,objectParams,hitSample

# osu 파일 전체 텍스트를 읽음
with open(osu_input, 'r', encoding='utf-8') as f:
    osu_text = f.read()

# HitObjects 섹션을 찾아 각 노트(히트 오브젝트)의 라인을 추출
hit_match = re.search(r"\[HitObjects\](.*)$", osu_text, re.S)
if not hit_match:
    raise ValueError("[HitObjects] 섹션을 찾을 수 없습니다.")

# HitObjects 섹션 이후의 각 줄을 가져와 노트 정보 라인만 리스트로 저장
hit_lines = [l for l in hit_match.group(1).strip().splitlines() if l.strip()]
print("노트가 할당된 모든 time의 수 : ", len(hit_lines))

# times_ms: 각 노트의 타임스탬프(밀리초)를 저장하는 리스트
# osu 채보 형식에서 콤마로 분리된 세 번째 값이 타임스탬프임
times_ms = [int(line.split(',')[2]) for line in hit_lines]

# Events 섹션 위치를 찾아두기 (나중에 새로운 이벤트를 삽입할 때 사용)
events_match = re.search(
    r"\[Events\](.*?)(?=\r?\n\[|$)",
    osu_text,
    re.S
)


"""
====================================================
================= 노트 기반 키음 추출 =================
====================================================
"""

# --- 노트 추출 함수 ---
def extract_clips(strategy_name, dominant_indices):
    """
    strategy_name: 적용한 스무딩(정제) 방법 이름 (문자열)
    dominant_indices: 각 노트 시간별로 우선 음을 가진 트랙 인덱스 리스트
    """
    print(f"\n▶ 추출 중: {strategy_name} 방식")
    output_dir = f"splited_notes_{strategy_name}"  # 결과 출력 디렉토리 이름
    os.makedirs(output_dir, exist_ok=True)  # 디렉토리 생성 (이미 있으면 넘어감)
    osu_output = f"{base_name}_{strategy_name}.osu"  # 생성할 새로운 osu 파일 이름

    # 노트 클립에 적용할 페이드 길이 계산 (fade_ms가 정의되어야 함)
    fade_len = int(sr * fade_ratio / 1000)  # 밀리초 -> 샘플 수로 변환
    # ramp: 0에서 1로 선형 증가하는 배열 (페이드 인에 사용)
    ramp = np.linspace(0, 1, fade_len)

    # silent_tracks: 원본 트랙들 복사본, 노트를 추출한 구간은 0으로 만들어서 반주 생성에 사용
    silent_tracks = [w.copy() for w in waves]
    new_events, new_hit_lines = [], []  # 새로운 osu 이벤트/히트오브젝트 정보를 담을 리스트
    counter = 1  # 추출된 노트 클립 번호 (이름 지정용)

    # 각 히트 라인(노트 시간)에 대해 반복
    for i, line in tqdm(enumerate(hit_lines), total=len(hit_lines), desc=f"Extracting ({strategy_name})"):
        parts = line.split(',')  # osu 채보 라인을 콤마로 분리
        time_ms = times_ms[i]    # 현재 노트의 시작 시간 (밀리초)

        # 다음 노트까지의 간격 또는 default_clip_ms 중 짧은 값으로 클립 길이 설정
        if i < len(times_ms) - 1:
            next_ms = times_ms[i + 1]
            # 두 노트 사이 간격, 기본값으로 설정된 길이(default_clip_ms) 중 작은 것을 선택
            clip_ms = min(default_clip_ms, next_ms - time_ms)
        else:
            clip_ms = default_clip_ms

        clip_len = int(sr * clip_ms / 1000)  # 클립 길이를 샘플 수로 계산
        fade_len = max(1, int(clip_len * fade_ratio))  # 최소 1샘플, 페이드 길이 재계산
        ramp = np.linspace(0, 1, fade_len)  # 페이드 인/아웃을 위한 선형 배열 재생성

        # dominant_indices에서 현재 노트에 해당하는 주도 트랙 인덱스를 구함
        idx = dominant_indices[i]
        start = int(sr * time_ms / 1000)  # 노트 시작점 샘플 인덱스
        end = min(start + clip_len, len(waves[0]))  # 클립 끝점 (오디오 길이 초과 방지)

        # 선택된 트랙에서 노트 오디오 클립을 복사
        clip = waves[idx][start:end].copy()
        if len(clip) > 2 * fade_len:
            # 클립 앞과 뒤에 페이드 인/아웃 적용
            clip[:fade_len] *= ramp         # 시작 부분 페이드 인
            clip[-fade_len:] *= ramp[::-1]  # 끝 부분 페이드 아웃
        else:
            # 클립이 너무 짧으면 전체에 걸쳐서 선형 페이드 적용
            clip *= np.linspace(0, 1, len(clip))

        # silent_tracks에서는 해당 노트 구간을 0으로 만들어서 반주용 트랙 생성에 사용
        silent_tracks[idx][start:end] = 0

        # 클립을 파일로 저장 (파일명: note_{트랙인덱스+1}_{카운터}.wav)
        clip_name = f"note_{idx+1}_{counter}.wav"
        sf.write(os.path.join(output_dir, clip_name), clip, sr)

        # osu 채보 행 끝의 샘플 파일 부분을 새 클립 이름으로 교체
        tail = parts[-1]
        effects, _ = (tail.rsplit(':', 1) if ':' in tail else (tail, ''))
        parts[-1] = f"{effects}:{clip_name}"
        new_hit_lines.append(','.join(parts))  # 새 히트오브젝트 정보로 저장
        new_events.append(f"Sample,{time_ms},0,\"{clip_name}\",50")  # Events 정보에 샘플 참조 추가
        counter += 1  # 다음 노트 번호로 증가

    # --- 반주 (accompaniment) 생성 ---
    # silent_tracks에 남은 소리들을 모두 더해서 반주용 오디오 생성
    accompaniment = np.zeros_like(waves[0])  # 초기값 0 배열 (원본 길이와 동일)
    for tr in silent_tracks:
        accompaniment += tr
    # 반주 오디오 파일로 저장
    sf.write(os.path.join(output_dir, "audio.wav"), accompaniment, sr)

    # --- 새로운 osu 파일 생성 ---
    # 기존 osu 텍스트에서 [Events] 시작부터 [HitObjects] 시작 전까지를 재구성
    header = osu_text[:events_match.start()]
    between = osu_text[events_match.end():hit_match.start()]
    footer = osu_text[hit_match.end():]
    with open(osu_output, 'w', encoding='utf-8') as f:
        f.write(header)
        f.write("[Events]\n")
        # 새로 생성된 샘플 이벤트를 파일에 추가
        for ev in new_events:
            f.write(ev + "\n")
        f.write("\n" + between)
        f.write("[HitObjects]\n")
        # 새로 생성된 히트오브젝트(노트) 정보 추가
        for hl in new_hit_lines:
            f.write(hl + "\n")
        f.write(footer)

    print(f"{strategy_name} 완료 - 파일 위치: {output_dir}, {osu_output}")




"""
====================================================
===================== 필터 설정 =====================
====================================================
"""

"""
smooth_none: 에너지 계산 결과를 그대로 사용 (변경 없음)
"""
def smooth_none(indices):
    return indices

"""
smooth_mode: window 기반 다수결 방식으로 'dominant_indices'를 정제
window 크기=5로 주변 5개 인덱스의 최빈값(majority) 선택
"""
def smooth_mode(indices, window=5):
    pad = window // 2
    padded = np.pad(indices, (pad, pad), mode='edge')
    # 중앙 부분을 중심으로 window 만큼 슬라이딩하며 최빈값 계산
    return [int(mode(padded[i:i+window], keepdims=False).mode) for i in range(len(indices))]

"""
smooth_median: median filter를 사용하여 돌출값(outlier)을 제거
"""
def smooth_median(indices, size=5):
    return median_filter(indices, size=size).tolist()

"""
smooth_priority: 우선순위와 에너지 임계값을 고려하여 dominant 인덱스 결정
- 같은 시간에 복수 노트(hit lines)가 있을 경우를 처리
"""
def smooth_priority(dominant_indices, *, times_ms, sr, waves, preferred_order, threshold=0.3):
    time_to_indices = {}
    for i, t in enumerate(times_ms):
        time_to_indices.setdefault(t, []).append(i)
    # 각 노트(인덱스)별로 결정된 dominant 트랙 인덱스를 담을 리스트
    dominant_by_index = [None] * len(times_ms)

    # 동일 시간에 있는 노트 그룹별로 처리
    for t, indices in time_to_indices.items():
        start = int(sr * t / 1000)
        end = start + int(sr * default_clip_ms / 1000)
        end = min(end, len(waves[0]))
        # 각 트랙의 해당 구간 에너지 계산
        energies = [np.sum(w[start:end] ** 2) for w in waves]
        total_energy = sum(energies)

        if len(indices) > 1:
            # 둘 이상의 노트가 같은 시간에 있을 때, 에너지 순으로 인덱스를 배정
            sorted_idx = sorted(range(len(energies)), key=lambda x: -energies[x])
            for j, i in enumerate(indices):
                dominant_by_index[i] = sorted_idx[j % len(waves)]
        else:
            # 노트가 하나만 있을 때는 우선순위를 고려하여 주도 트랙 선택
            chosen = None
            for p in preferred_order:
                # 특정 트랙이 전체 에너지의 threshold 이상이면 그 트랙 선택
                if energies[p] >= threshold * total_energy:
                    chosen = p
                    break
            if chosen is None:
                # 임계값을 만족하는 트랙이 없으면 에너지 최대인 트랙 선택
                chosen = int(np.argmax(energies))
            dominant_by_index[indices[0]] = chosen

    return dominant_by_index


"""
====================================================
======================= 설정 =======================
====================================================
"""
# --- base dominant index 계산 (기본 방식) ---
base_dominant_indices = []
for i, time_ms in enumerate(times_ms):
    start = int(sr * time_ms / 1000)
    if i < len(times_ms) - 1:
        next_ms = times_ms[i+1]
        clip_ms = min(default_clip_ms, next_ms - time_ms)
    else:
        clip_ms = default_clip_ms
    end = min(start + int(sr * clip_ms / 1000), len(waves[0]))
    # 각 트랙의 해당 구간 에너지 계산
    energies = [np.sum(w[start:end]**2) for w in waves]
    # 가장 높은 에너지 값을 가진 트랙의 인덱스를 선택
    idx = int(np.argmax(energies))
    base_dominant_indices.append(idx)


"""
smooth_base_normalized:
- drum(인덱스=2)과 other(인덱스=3) 트랙은 전체 에너지 대비 상대적 비율로 스케일 조정
  (패턴 인식 시 드럼/기타가 가진 전체 비중을 계산하여 보정)
- 나머지 트랙은 원래 방식대로 계산
"""
def smooth_base_normalized(times_ms, sr, waves):
    drum_idx, other_idx = 2, 3
    # drum과 other 트랙의 전체 에너지
    drum_energy_max = np.sum(waves[drum_idx]**2)
    other_energy_max = np.sum(waves[other_idx]**2)

    result = []
    for i, time_ms in enumerate(times_ms):
        start = int(sr * time_ms / 1000)
        if i < len(times_ms) - 1:
            next_ms = times_ms[i+1]
            clip_ms = min(default_clip_ms, next_ms - time_ms)
        else:
            clip_ms = default_clip_ms
        end = min(start + int(sr * clip_ms / 1000), len(waves[0]))

        energies = [np.sum(w[start:end] ** 2) for w in waves]

        # drum 트랙은 전체 drum 에너지 대비 비율로 스케일 보정
        if drum_energy_max > 0:
            energies[drum_idx] = energies[drum_idx] / drum_energy_max * 100
        # other 트랙도 마찬가지
        if other_energy_max > 0:
            energies[other_idx] = energies[other_idx] / other_energy_max * 100

        # 가장 큰 에너지를 가지는 트랙 인덱스 선택
        idx = int(np.argmax(energies))
        result.append(idx)
    return result

# 각 스무딩 방법을 딕셔너리로 정의
smoothing_methods = {
    "base": smooth_none,
    "mode": smooth_mode,
    "median": smooth_median,
    "priority": partial(
        smooth_priority,
        times_ms=times_ms,
        sr=sr,
        waves=waves,
        preferred_order=preferred_order,
        threshold=energy_threshold_ratio
    ),
    "base_normalized": lambda _: smooth_base_normalized(times_ms, sr, waves)
}

"""
====================================================
======================= 실행 =======================
====================================================
"""

# --- smoothing 방식 적용 및 노트 추출 ---
for name, func in smoothing_methods.items():
    # base_dominant_indices 리스트를 스무딩(정제) 함수에 입력하여 새로운 dominant index 리스트 생성
    smoothed_indices = func(base_dominant_indices)
    # 생성된 인덱스를 사용하여 extract_clips 함수로 노트 오디오 추출
    extract_clips(name, smoothed_indices)