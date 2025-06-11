import os
import librosa
import soundfile as sf
import numpy as np
import re
from tqdm import tqdm
from collections import defaultdict

# --- 설정 ---
wav_paths = {
    2: "unwelcomeSchool/drums.wav",
    3: "unwelcomeSchool/other.wav"
}

track_gains = {
    2: 1.0,
    3: 1.75
}

x_positions = {
    2: 64,
    3: 320
}

onset_params = {
    2: {"hop_length": 333, "delta": 0.1},
    3: {"hop_length": 333, "delta": 0.025}
}

osu_input = "UnwelcomeSchool_Hard.txt"
base_name = os.path.splitext(os.path.basename(osu_input))[0]
osu_output = f"{base_name}_onset.osu"
output_dir = f"splited_notes_onset"
os.makedirs(output_dir, exist_ok=True)

# --- osu 파일 파싱 (헤더/푸터 분리) ---
with open(osu_input, 'r', encoding='utf-8') as f:
    osu_text = f.read()

events_match = re.search(r"\[Events\]", osu_text)
header = osu_text[:events_match.start()]

hitobj_match = re.search(r"\[HitObjects\]", osu_text)
footer = osu_text[hitobj_match.start():]

# --- 1) Onset Detection & 노트 리스트 생성 ---
notes = []  # (track_idx, onset_ms)
sr = None
track_to_onsets = defaultdict(list)

for idx, path in wav_paths.items():
    y, sr = librosa.load(path, sr=None)
    y *= track_gains.get(idx, 1.0)

    params = onset_params[idx]
    onsets = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        units="time",
        hop_length=params["hop_length"],
        delta=params["delta"],
        normalize=True,
        sparse=True,
        backtrack=True,
    )
    onsets_ms = (onsets * 1000).astype(int)

    for t in onsets_ms:
        notes.append((idx, t))
        track_to_onsets[idx].append(t)

# 각 트랙의 마지막 구간을 끝까지 자르기 위해 종료 시점 추가
for idx in track_to_onsets:
    y_full, _ = librosa.load(wav_paths[idx], sr=sr)
    total_ms = int(len(y_full) / sr * 1000)
    track_to_onsets[idx].append(total_ms)

# 마지막 onset 구간도 notes에 포함
for idx, onsets in track_to_onsets.items():
    last_onset = onsets[-2]  # 마지막 onset (마지막은 종료시간이니까)
    if (idx, last_onset) not in notes:
        notes.append((idx, last_onset))

# 시간 순 정렬
notes.sort(key=lambda x: x[1])

# --- 2) 노트 클립 자르기 & osu 이벤트 생성 ---
new_events = []
new_hitobjects = []
counter = 1

for idx, t in tqdm(notes, desc="Generating note clips"):
    y_wave, _ = librosa.load(wav_paths[idx], sr=sr)
    y_wave *= track_gains.get(idx, 1.0)

    lst = track_to_onsets[idx]
    i = lst.index(t)
    t_next = lst[i + 1]
    clip_ms = t_next - t

    start = int(sr * t / 1000)
    end = min(int(sr * (t + clip_ms) / 1000), len(y_wave))
    clip = y_wave[start:end]

    clip_name = f"note_onset_{idx}_{counter}.wav"
    sf.write(os.path.join(output_dir, clip_name), clip, sr)

    new_events.append(f'Sample,{t},0,"{clip_name}",50')

    x = x_positions[idx]
    y = 192
    type_ = 1
    hitSound = 0
    objectParams = "0:0:0:0"
    hitSample = clip_name
    new_hitobjects.append(f"{x},{y},{t},{type_},{hitSound},{objectParams},{hitSample}")

    counter += 1

# --- 3) osu 파일 쓰기 ---
with open(osu_output, 'w', encoding='utf-8') as f:
    f.write(header)
    f.write("[Events]\n")
    for e in new_events:
        f.write(e + "\n")
    f.write("\n")
    f.write("[HitObjects]\n")
    for ho in new_hitobjects:
        f.write(ho + "\n")

print("완료:")
print(f" - Notes detected: {len(new_hitobjects)}")
print(f" - Audio clips → {output_dir}")
print(f" - Osu file → {osu_output}")
