#!/usr/bin/env bash
# =============================================================================
# 이 폴더를 GitHub (기본: https://github.com/niceread102-cmyk) 에 커밋 + 푸시
#
# 사용법 (이 폴더에서):
#   ./upload_to_github.sh --dry-run                    # 올라갈 파일 목록과 크기만 확인 (아무것도 안 올림)
#   ./upload_to_github.sh                              # 커밋 + 푸시 (토큰은 물어봄)
#   ./upload_to_github.sh "커밋 메시지"                  # 메시지를 직접 지정
#   GITHUB_TOKEN=ghp_xxx ./upload_to_github.sh         # 토큰을 환경변수로
#
# 설정 (환경변수로 바꿀 수 있음):
#   GITHUB_OWNER  기본 niceread102-cmyk  (조직에 올리려면 예: GITHUB_OWNER=f2x-project)
#   REPO_NAME     기본 이 폴더 이름 (direction_recognition_sideplate_v3)
#   VISIBILITY    private(기본) | public   — 저장소를 새로 만들 때만 적용
#   BRANCH        기본 main
#   REMOTE        기본 personal  — 기존 origin(f2x-project) 연결은 건드리지 않고 별도 remote 로 푸시
#   GIT_NAME / GIT_EMAIL   이 저장소에만 쓸 커밋 작성자 (전역 git 설정은 건드리지 않음)
#   GITHUB_TOKEN  GitHub Personal Access Token
#                 - classic: repo 권한 / fine-grained: Administration(만들기) + Contents(푸시) 쓰기
#                 - 토큰은 .git/config 나 명령줄 인자(ps 로 보임)에 남기지 않는다.
#   USE_SSH=1     SSH 키로 푸시 (토큰 불필요, 저장소는 GitHub 에서 미리 만들어 두어야 함)
#
# 올리지 않는 것: .gitignore 참고 (캐시, 원본 촬영 사진, 디버그·시험 그림, 백업, 로그)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

GITHUB_OWNER="${GITHUB_OWNER:-${GITHUB_ORG:-niceread102-cmyk}}"
REPO_NAME="${REPO_NAME:-$(basename "$PWD")}"
VISIBILITY="${VISIBILITY:-private}"
BRANCH="${BRANCH:-main}"
REMOTE="${REMOTE:-personal}"
USE_SSH="${USE_SSH:-0}"
DRY_RUN=0
COMMIT_MSG="${COMMIT_MSG:-}"
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
        *)         COMMIT_MSG="$arg" ;;
    esac
done
COMMIT_MSG="${COMMIT_MSG:-side_plate 회전 방향 인식 업데이트 ($(date +%Y-%m-%d))}"

say()  { printf '\033[1;34m[upload]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[upload] 오류:\033[0m %s\n' "$*" >&2; exit 1; }

# GitHub API 호출: 토큰을 stdin 으로 curl 에 넘겨 명령줄에 드러나지 않게 한다.
# 출력: 응답 본문 + 마지막 줄에 HTTP 코드
gh_api() {  # gh_api METHOD PATH [JSON_BODY]
    local method="$1" path="$2" body="${3:-}"
    local args=(-s -X "$method" -w '\n%{http_code}' -K -
                -H "Accept: application/vnd.github+json"
                -H "X-GitHub-Api-Version: 2022-11-28")
    [[ -n "$body" ]] && args+=(-d "$body")
    printf 'header = "Authorization: Bearer %s"\n' "$GITHUB_TOKEN" \
        | curl "${args[@]}" "https://api.github.com$path"
}

# ---- 1. 로컬 저장소 준비 -----------------------------------------------------
if [[ ! -d .git ]]; then
    say "git 저장소를 만듭니다 (브랜치 $BRANCH)"
    git init -q -b "$BRANCH"
fi
[[ -n "${GIT_NAME:-}"  ]] && git config user.name  "$GIT_NAME"
[[ -n "${GIT_EMAIL:-}" ]] && git config user.email "$GIT_EMAIL"
cur_branch=$(git symbolic-ref --short HEAD 2>/dev/null || echo "$BRANCH")
[[ "$cur_branch" == "$BRANCH" ]] || fail "현재 브랜치가 $cur_branch 입니다. BRANCH=$cur_branch 로 실행하거나 $BRANCH 로 바꾸세요."

# ---- 2. 올라갈 파일 확인 ------------------------------------------------------
git add -A
say "변경된 파일: $(git diff --cached --name-only | wc -l)개, 저장소 전체 $(git ls-files | wc -l)개"
total=$(git ls-files -z | xargs -0 du -cb 2>/dev/null | tail -1 | cut -f1)
say "올라갈 파일 합계: $(numfmt --to=iec "$total" 2>/dev/null || echo "$total B")"
big=$(git ls-files -z | xargs -0 -I{} find {} -maxdepth 0 -size +50M 2>/dev/null || true)
[[ -n "$big" ]] && fail "50 MB 넘는 파일이 있습니다 (GitHub 경고/거부 대상): $big"
if [[ $DRY_RUN -eq 1 ]]; then
    git status --short | sed 's/^/    /'
    if git rev-parse -q --verify HEAD >/dev/null; then git reset -q; else git rm -r -q --cached . >/dev/null; fi
    say "--dry-run: 대상 https://github.com/$GITHUB_OWNER/$REPO_NAME ($BRANCH, remote '$REMOTE')"
    say "--dry-run: 커밋/푸시하지 않았습니다."
    exit 0
fi

# ---- 3. 커밋 ---------------------------------------------------------------
say "커밋 작성자: $(git config user.name || echo '?') <$(git config user.email || echo '?')>  (바꾸려면 GIT_NAME / GIT_EMAIL)"
if git diff --cached --quiet; then
    say "새로 커밋할 변경이 없습니다. 기존 커밋만 푸시합니다."
else
    git commit -q -m "$COMMIT_MSG"
    say "커밋: $(git log -1 --oneline)"
fi

# ---- 4. GitHub 저장소 확인 / 만들기 -------------------------------------------
if [[ "$USE_SSH" != "1" ]]; then
    if [[ -z "${GITHUB_TOKEN:-}" ]]; then
        [[ -t 0 ]] || fail "GITHUB_TOKEN 이 필요합니다. 예) GITHUB_TOKEN=ghp_xxx ./upload_to_github.sh"
        read -rsp "GitHub 토큰 (입력은 화면에 안 보임): " GITHUB_TOKEN; echo
        [[ -n "$GITHUB_TOKEN" ]] || fail "토큰이 비어 있습니다."
    fi

    resp=$(gh_api GET /user); rc=$(tail -1 <<<"$resp")
    [[ "$rc" == "200" ]] || fail "토큰이 유효하지 않습니다 (HTTP $rc)."
    login=$(head -n -1 <<<"$resp" | grep -m1 '"login"' | sed -E 's/.*"login": *"([^"]+)".*/\1/')
    say "GitHub 로그인: $login"

    rc=$(gh_api GET "/repos/$GITHUB_OWNER/$REPO_NAME" | tail -1)
    if [[ "$rc" == "404" ]]; then
        say "GitHub 에 $GITHUB_OWNER/$REPO_NAME 저장소를 만듭니다 ($VISIBILITY)"
        private=true; [[ "$VISIBILITY" == "public" ]] && private=false
        body="{\"name\":\"$REPO_NAME\",\"private\":$private,\"description\":\"side_plate 빨간 홈/문양 방향 인식 (STL 맞추기 + Photoneo 2D)\"}"
        if [[ "${login,,}" == "${GITHUB_OWNER,,}" ]]; then
            resp=$(gh_api POST /user/repos "$body")
        else
            resp=$(gh_api POST "/orgs/$GITHUB_OWNER/repos" "$body")
        fi
        rc=$(tail -1 <<<"$resp")
        [[ "$rc" == "201" ]] || fail "저장소를 만들지 못했습니다 (HTTP $rc): $(head -n -1 <<<"$resp" | head -c 300)"
    elif [[ "$rc" == "200" ]]; then
        say "GitHub 에 $GITHUB_OWNER/$REPO_NAME 저장소가 이미 있습니다. 그대로 푸시합니다."
    else
        fail "저장소 확인 실패 (HTTP $rc). 토큰 권한과 계정 이름($GITHUB_OWNER)을 확인하세요."
    fi
fi

# ---- 5. 푸시 ---------------------------------------------------------------
url="https://github.com/$GITHUB_OWNER/$REPO_NAME.git"
[[ "$USE_SSH" == "1" ]] && url="git@github.com:$GITHUB_OWNER/$REPO_NAME.git"
if git remote get-url "$REMOTE" >/dev/null 2>&1; then
    git remote set-url "$REMOTE" "$url"
else
    git remote add "$REMOTE" "$url"
fi
say "푸시: $url ($BRANCH → remote '$REMOTE')"
if [[ "$USE_SSH" == "1" ]]; then
    git push -u "$REMOTE" "$BRANCH"
else
    # 토큰은 환경변수로 이 push 에만 넘기고 .git/config / 명령줄에는 남기지 않는다
    auth=$(printf 'x-access-token:%s' "$GITHUB_TOKEN" | base64 -w0)
    GIT_TERMINAL_PROMPT=0 GIT_CONFIG_COUNT=1 \
    GIT_CONFIG_KEY_0="http.https://github.com/.extraHeader" \
    GIT_CONFIG_VALUE_0="Authorization: Basic $auth" \
        git push -u "$REMOTE" "$BRANCH"
fi
say "완료: https://github.com/$GITHUB_OWNER/$REPO_NAME"
