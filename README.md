# Developer Rules & Constraints for cb_t02 Project

1. 모든 코드 파일은 200줄 이하로 유지해야 합니다. 200줄을 초과하면 책임 단위로 파일을 분할하세요.
2. 각 함수는 100줄 이하로 유지해야 합니다. 필요하면 리팩토링하고 분할하세요.
3. Windows 10 + RTX 4060 (8GB) 환경에서 PyTorch 2.3.0(GPU Only)과 torchtext 0.18.0은 Windows 환경에서 CUDA 11.8을 지원하는 마지막 버전입니다. 및 호환 가능한 라이브러리만 사용하세요. 최신 버전이나 호환되지 않는 패키지는 사용하지 마세요.
4. 모든 데이터셋은 /datas/에서 로드해야 하며, 모든 모델(임베딩 포함)은 /models/에서 로드해야 합니다. (CPU버전 절대금지)
Docker, WSL, 또는 기타 가상화 환경을 사용하지 마세요. 프로젝트는 Windows에서 네이티브로 실행되어야 합니다.
5. 폴더 및 파일 명명 시 모듈화와 미래 확장성을 항상 유지하세요.
6. 메인 브랜치에 사용하지 않거나 테스트용 파일을 추가하거나 남겨두지 마세요.
7. 모든 핵심 모듈에 충분한 예외 처리와 로깅을 추가하세요.
8. 테스트 코드와 메인 코드를 엄격히 분리하세요.
9. 모든 코드, 도큐스트링, 변수명은 영어만 사용하세요. (단, 주석은 한글로만 작성할것)
10. 지침을 최우선적으로 따르며 사용자가 특별하게 지시하면 그것을 먼저 따르도록.
11. 깃 저장소 내 기존 브랜치 및 커밋 히스토리를 참고하여 전체 프로젝트 진행상황과 변경사항을 파악

    - 파일 간 의존성 최소화, 인터페이스 활용, 순환참조 금지.
    - 설정/상수는 config 파일로만 관리.
    - 실행/테스트 코드는 분리.
    - 역할별 폴더/모듈 구조 유지, cross-import 금지.

    - 작업 전 전체 구조/경로/지침 상기.
    - 출력 후 지침 위반 여부 자체 점검.
    - 위반 시 즉시 전체 코드 재출력.

    - PEP8 준수, 타입 힌트, docstring 필수.
    - 외부 패키지 주석 설명.

12. 데이터셋 JSON 양식 설명
    question: 질문 관련 정보    
    text: 질문 원문 문자열    
    tokens: 질문 문장의 토큰별 형태소 및 품사 정보 배열 
    각 토큰은 다음 필드 포함    
    text: 토큰 텍스트    
    lemma: 토큰 기본형(어간)    
    pos: 국립국어원 표준 품사 태그 (예: NNG, VV, JKS 등)    
    answer: 답변 관련 정보 (question과 동일 구조)    
    concepts: 질문과 답변에서 추출한 핵심 개념 또는 키워드 배열 (문자열 목록)    
    domain: 대화쌍이 속하는 주제 분야(예: 기술, 역사, 문화 등) 문자열

    코드 수정/분할 시 import·참조 관계를 구조에 맞게 동기화.
    예외처리, 디버깅, 로깅, 테스트 코드 포함 필수.

**If any violation is found, immediately refactor and push again.**

## Environment Variables

- `PYTHONIOENCODING` - default `utf-8`
- `SOYNLP_DATA_DIR` - optional path for soynlp cache

Use `scripts/start_dev.sh` to launch the API server during development.
