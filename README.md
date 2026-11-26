# LLM intent, 메타데이터, 최근 이력이 추천 성능에 미치는 영향

Goodreads Children과 MovieLens 25M에서 협업 필터링에 세 신호를 더하고, 추천 정확도와 롱테일 노출이 어떻게 달라지는지 평가하는 졸업작품 소스코드다. `reproduce.py`에서 데이터셋을 고르면 원본 다운로드부터 학습 구간 프로필 생성, Luxia API 호출, 지식 그래프(KG) 구성, 모델 학습, 사용자별 평가와 보고서 작성까지 실행한다.

## 설치

Python 3.11 가상환경에서 저장소 루트를 작업 디렉터리로 사용한다. 두 실험은 같은 환경을 쓴다. 다음은 Windows PowerShell과 NVIDIA GPU(CUDA 12.1용 PyTorch) 기준이다.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

Linux에서는 `python3.11 -m venv .venv`, `source .venv/bin/activate`로 환경을 연 뒤 같은 설치 명령을 사용한다. CPU에서 합성 데이터 테스트를 하려면 PyTorch 인덱스를 `https://download.pytorch.org/whl/cpu`로 바꾼다. 실제 데이터의 전체 모델 비교에는 GPU를 권장하며, Goodreads 전용 intent 필터 실험은 CUDA가 필요하다.

## 공식 데이터와 API 키

| 데이터셋 | 다운로드 파일 | k=30 필터 적용 후 사용자 / 아이템 / 상호작용 |
|---|---|---:|
| Goodreads Children | 아동 도서 상호작용, 도서 메타데이터, 전체 저자 메타데이터 JSON.gz | 62,142 / 27,975 / 6,237,437 |
| MovieLens 25M | `ml-25m.zip`의 `ratings.csv`와 `movies.csv` | 133,684 / 15,785 / 24,046,865 |

다운로드할 때 파일 크기와 SHA-256을 확인한다. Goodreads 파일은 `experiments/goodreads/`에, 영화 CSV는 `experiments/movielens/raw/ml-25m/`에 저장한다. 같은 원본이 있으면 재사용한다. 원본 데이터에는 배포처의 이용 조건이 적용된다. [UCSD Goodreads Book Graph](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html) · [GroupLens MovieLens 25M](https://grouplens.org/datasets/movielens/25m/) · [MovieLens 압축 파일 SHA-256 출처](https://raw.githubusercontent.com/tensorflow/datasets/master/tensorflow_datasets/url_checksums/movielens.txt)

**API 키 없이 준비하고 비용을 추정하려면** 아래 두 명령을 실행한다. 다운로드만 하려면 `prepare`를 `download`로 바꾼다.

```powershell
python reproduce.py goodreads prepare
python reproduce.py movielens prepare
```

**실제 API를 사용하려면** 해당 실험 폴더의 `.env.example`을 같은 폴더의 `.env`로 복사하고 `LUXIA_API_KEY` 값을 입력한다. 두 실험에서 같은 키를 쓸 때는 환경변수 `LUXIA_API_KEY` 하나로도 설정할 수 있다. API에는 각 데이터셋의 학습 이력에서 만든 프로필 텍스트와 아이템 프로필이 전달된다. 키와 원본 데이터, 생성한 캐시는 Git에서 제외된다.

## 보고서 재현

```powershell
python reproduce.py goodreads run --split TO
python reproduce.py movielens run --split TO
python reproduce.py movielens run --split TO_GLOBAL
```

각 명령은 빈 폴더 상태에서도 필요한 데이터를 받고, 이미 완료한 준비 단계와 유효한 캐시를 재사용한다. 호출 수와 예상 비용을 확인하고 `y`를 입력하면 연결 확인용 요청을 보낸 뒤 본 호출을 시작한다. `--max-usd 50`은 **예상 비용의 상단**이 50달러를 넘을 때 시작을 막는다. 실제 청구 중단은 제공자 측 한도 설정에 따른다. 첫 RAG 실행에는 Hugging Face의 `sentence-transformers/all-mpnet-base-v2` 모델 다운로드가 필요하다.

| 단계 | 처리 | 주요 산출물 |
|---|---|---|
| 0 · A · T | 상호작용 변환, 사용자·아이템 30-core, 시간순 분할용 RecBole 데이터 생성 | `data/kc_k30/ikgr-custom/ikgr-custom.inter` |
| P · S | 학습 이력으로 사용자 프로필 작성, 캐시를 반영해 비용 추정 | `data/kc_k30/profiles_k30_<분할>.csv`, `llm_cost_estimate.json` |
| B · C | API로 intent 추출, 인코더가 찾은 후보에서 관련 intent 선택 | `step1_intents.csv`, `step2_related_intents.csv` |
| D | intent 및 아이템 메타데이터 그래프 구축 | `kg_pack.pt`, `meta_kg_pack.pt` |
| E · R | 학습 데이터 누수 검사, full-sort 평가, 사용자별 paired bootstrap과 보고서 | `slice_eval_<분할>_result.json`, `per_user_<분할>/`, `report_<분할>.md` |

산출물은 각 실험 아래 `run_k30_TO/` 또는 `run_k30_TO_GLOBAL/`에 생긴다. `TO`는 사용자별 시간순 80/10/10, `TO_GLOBAL`은 전체 시간순 70/10/20이다. MovieLens의 `TO_GLOBAL`은 학습 이력이 있는 사용자로 조기 종료 시점을 고르고, 해당 사용자의 결과를 `report_TO_GLOBAL_warm.md`에도 쓴다. Goodreads 원 보고서의 주 분할은 `TO`다.

기본 비교는 MF, BPR, LightGCN과 두 백본의 메타데이터·recency 조합 및 intent 추가 조합까지 총 7개 모델이다. 시드는 2020·2021·2022, 임베딩은 64차원, 학습은 최대 40 epoch이다. 한 시드만 실행하려면 `--seeds 2020`을 붙인다. 지표에는 NDCG, Recall, Hit, tail Recall, Coverage가 포함된다.

### 원 보고서와 수치 대조

새 API 응답으로 만든 intent는 같은 프롬프트에서도 달라질 수 있다. **공개된 intent를 고정하고 학습만 재현**할 때는 원본 커밋의 해당 `data_bundle/` 내용을 같은 실험 폴더의 `data_bundle/`로 복사한 뒤, 아래 명령으로 원 저자의 체크섬 검사를 거쳐 푼다. 새로 생성하던 실험과 분리된 체크아웃을 사용한다. 이미 D 단계까지 마친 환경에서는 바로 `evaluate`를 실행할 수 있다.

```powershell
# 저장소 루트에서 실행
python experiments/goodreads/scripts/unpack_data_bundle.py
python reproduce.py goodreads evaluate --split TO
python experiments/movielens/scripts/unpack_data_bundle.py --split TO
python reproduce.py movielens evaluate --split TO
python experiments/movielens/scripts/unpack_data_bundle.py --split TO_GLOBAL
python reproduce.py movielens evaluate --split TO_GLOBAL
```

- **Goodreads**의 원 보고서는 여러 실행을 합쳤다. LightGCN+intent의 발표 수치는 seed 2020이며 통합 명령은 세 시드를 모두 평가한다. 원 보고서의 MF·intent 제거율 표는 아래 전용 러너의 설정을 사용한다. [고정된 원 보고서](https://github.com/HK1DS/goodreads_november/blob/ef1d2d7d7e3b7c7c65963b0e4ee6964a08ac5cf0/report.md) · [평가 입력 번들](https://github.com/HK1DS/goodreads_november/tree/ef1d2d7d7e3b7c7c65963b0e4ee6964a08ac5cf0/data_bundle)
- **MovieLens**의 발표 표는 각 분할에서 7개 모델 × 3개 시드의 평균이다. `TO`에서 MF의 test NDCG@10은 0.0876, 메타데이터와 recency를 함께 쓴 MF는 0.0892였다. [고정된 원 보고서](https://github.com/HK1DS/movielens_november/blob/10ecfee14f7a9c809bfa21de7b9a2e8d5e63c819/RESULTS.md) · [평가 입력 번들](https://github.com/HK1DS/movielens_november/tree/10ecfee14f7a9c809bfa21de7b9a2e8d5e63c819/data_bundle)

### 세부 실험

`TO`의 그래프와 평가 입력을 준비한 후 실행한다. 상호작용 채널을 하나씩 켜는 ablation은 데이터셋마다 2개 조건 × 3개 시드를 사용한다.

```powershell
cd experiments/goodreads
python scripts/run_todo_5_8.py --run --phase all
python scripts/run_ablation_nov22.py all
python analyze_user_segments.py
cd ../movielens
python scripts/run_ablation_nov22.py all
python scripts/analyze_activity_ablation.py pair --label "MovieLens MF" --off_dir run_k30_TO/per_user_TO --off IKGR_kgoff --on IKGR_dyn_nointent --out analysis/activity_TO/movielens_mf
```

## 테스트

각 실험은 전용 모델·어댑터 모듈을 같은 이름으로 가져오므로 아래처럼 별도 Python 프로세스로 검사한다. 스모크 테스트는 합성 데이터, 로컬 모의 API 서버와 모의 인코더를 사용한다. 두 시간 분할을 학습·평가하고, 캐시 재개와 누수 차단까지 확인한다. 테스트가 지정한 작업 폴더를 초기화하므로 새 임시 경로를 사용한다.

```powershell
$env:PYTHONUTF8 = "1"
python -m pytest -q tests
cd experiments/goodreads
python -m pytest -q tests
python -X utf8 tests/smoke_e2e.py --workdir "$env:TEMP/goodreads-smoke-$([guid]::NewGuid())"
cd ../movielens
python -m pytest -q tests
python -X utf8 tests/smoke_e2e.py --workdir "$env:TEMP/movielens-smoke-$([guid]::NewGuid())"
```

## 코드 출처

데이터별 코드·설정·프롬프트·테스트는 아래 커밋에서 가져왔다. Goodreads는 최근 읽은 책·저자·shelf, MovieLens는 평점이 높은 영화와 장르 분포로 학습 프로필을 만든다. 통합 실행기는 데이터 검증과 보고서 생성을 연결하며, 두 실행기에서 비용 확인을 연결 확인용 API 요청 앞으로 옮겼다. [Goodreads 소스](https://github.com/HK1DS/goodreads_november/tree/ef1d2d7d7e3b7c7c65963b0e4ee6964a08ac5cf0) · [MovieLens 소스](https://github.com/HK1DS/movielens_november/tree/10ecfee14f7a9c809bfa21de7b9a2e8d5e63c819)
