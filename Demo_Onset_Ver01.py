import os
import librosa
import soundfile as sf
import numpy as np
import re

# --- 설정 ---
wav_paths = {
    2: "unwelcomeSchool/drums.wav",  # track index → 파일 경로
    3: "unwelcomeSchool/other.wav"
}
# osu 배치할 x 좌표 (사이드 4레인 중 두 레인만 사용)
x_positions = {
    2: 64,
    3: 192
}

osu_input = "UnwelcomeSchool_Hard.txt"
base_name = os.path.splitext(os.path.basename(osu_input))[0]
osu_output = f"{base_name}_onset.osu"
output_dir = f"splited_notes_onset"
os.makedirs(output_dir, exist_ok=True)

# 기본 자르기 최소 길이 (ms)
min_clip_ms = 50
default_clip_ms = 333

# --- osu 파일 파싱 (헤더·푸터 분리) ---
with open(osu_input, 'r', encoding='utf-8') as f:
    osu_text = f.read()
# 헤더 (Events 앞까지)
events_match = re.search(r"\[Events\]", osu_text)
header = osu_text[:events_match.start()]

# HitObjects 시작 위치
hitobj_match = re.search(r"\[HitObjects\]", osu_text)
footer = osu_text[hitobj_match.start():]

# --- 1) Onset Detection & 노트 리스트 생성 ---
notes = []  # (track_idx, onset_ms)
sr = None

for idx, path in wav_paths.items():
    y, sr = librosa.load(path, sr=None)
    # 온셋 프레임 인덱스 검출
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")  # seconds
    onsets_ms = (onsets * 1000).astype(int)
    # 트랙별 onset 저장
    for t in onsets_ms:
        notes.append((idx, t))
# 시간 순 정렬
notes.sort(key=lambda x: x[1])

# --- 2) 노트 클립 잘라 저장 & osu 이벤트/히트오브젝트 생성 ---
new_events = []
new_hitobjects = []

# 트랙별 다음 온셋 시간을 구하기 위해 그룹핑
from collections import defaultdict
track_to_onsets = defaultdict(list)
for idx, t in notes:
    track_to_onsets[idx].append(t)

# 각 트랙의 onset 리스트에 끝점(마지막+default) 추가
for idx, lst in track_to_onsets.items():
    # 마지막 구간용 더미
    lst.append(lst[-1] + default_clip_ms if 'default_clip_ms' in globals() else lst[-1] + 333)

counter = 1
for idx, t in notes:
    # 원본 파형을 먼저 불러옵니다 (sr은 이미 파악돼 있으니 두 번째 리턴값 _ 무시)
    y_wave, _ = librosa.load(wav_paths[idx], sr=sr)

    # 다음 Onset 시각을 구하고, clip_ms 결정(밀리초)
    lst = track_to_onsets[idx]
    i = lst.index(t)
    t_next = lst[i+1]
    clip_ms = max(min_clip_ms, t_next - t)

    # 잘라낼 샘플 구간의 시작/끝 인덱스 계산
    start = int(sr * t / 1000)
    end   = min(int(sr * (t + clip_ms) / 1000), len(y_wave))

    clip = y_wave[start:end]  # 올바른 파형 배열에서 슬라이스

    # 파일 저장
    clip_name = f"note_onset_{idx}_{counter}.wav"
    sf.write(os.path.join(output_dir, clip_name), clip, sr)

    # osu [Events] 라인: Sample,<time>,0,"<파일명>",Volume
    new_events.append(f'Sample,{t},0,"{clip_name}",50')
    # osu [HitObjects] 라인 구성
    x = x_positions[idx]
    y = 192       # 고정 y 좌표 (임의)
    type_ = 1     # Circle
    hitSound = 0
    objectParams = "0:0:0:0"
    hitSample = clip_name
    new_hitobjects.append(f"{x},{y},{t},{type_},{hitSound},{objectParams},{hitSample}")

    counter += 1

# --- 3) osu 파일 쓰기 ---
with open(osu_output, 'w', encoding='utf-8') as f:
    # 헤더
    f.write(header)
    f.write("[Events]\n")
    # 새 이벤트들
    for e in new_events:
        f.write(e + "\n")
    f.write("\n")
    # 히트오브젝트 블록
    f.write("[HitObjects]\n")
    for ho in new_hitobjects:
        f.write(ho + "\n")

print("완료:")
print(" - Notes detected:", len(new_hitobjects))
print(" - Audio clips →", output_dir)
print(" - Osu file →", osu_output)
