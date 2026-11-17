#!/usr/bin/env bash
# =============================================================================
# direction_recognition_sideplate_v2 → GitHub (https://github.com/f2x-project) 올리기
#
# 사용법 (이 폴더에서):
#   ./upload_to_github.sh --dry-run                 # 올라갈 파일 목록과 크기만 확인 (아무것도 안 올림)
#   GITHUB_TOKEN=ghp_xxx ./upload_to_github.sh      # 저장소 만들기(없으면) + 커밋 + 푸시
#
# 설정 (환경변수로 바꿀 수 있음):
#   GITHUB_ORG    기본 f2x-project
#   REPO_NAME     기본 direction_recognition_sideplate_v2
#   VISIBILITY    private(기본) | public
#   BRANCH        기본 main
#   GIT_NAME / GIT_EMAIL   이 저장소에만 쓸 커밋 작성자 (전역 git 설정은 건드리지 않음)
#   COMMIT_MSG    커밋 메시지
#   GITHUB_TOKEN  GitHub Personal Access Token (repo 권한, 조직 저장소를 만들려면 조직 권한도)
#                 - 토큰은 .git/config 에 저장하지 않고 푸시할 때만 쓴다.
#                 - SSH 키가 있으면 USE_SSH=1 로 토큰 없이 푸시할 수 있다 (저장소는 미리 만들어 두어야 함).
#
# 올리지 않는 것: .gitignore 참고 (캐시, 원본 촬영 사진, 디버그·시험 그림, 백업, 로그)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

GITHUB_ORG="${GITHUB_ORG:-f2x-project}"
REPO_NAME="${REPO_NAME:-direction_recognition_sideplate_v2}"
VISIBILITY="${VISIBILITY:-private}"
BRANCH="${BRANCH:-main}"
COMMIT_MSG="${COMMIT_MSG:-side_plate 회전 방향 인식 v2 (Hikvision + Photoneo 2D)}"
USE_SSH="${USE_SSH:-0}"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

say()  { printf '\033[1;34m[upload]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[upload] 오류:\033[0m %s\n' "$*" >&2; exit 1; }

# ---- 1. 로컬 저장소 준비 -----------------------------------------------------
if [[ ! -d .git ]]; then
    say "git 저장소를 만듭니다 (브랜치 $BRANCH)"
    git init -q -b "$BRANCH"
fi
if [[ -n "${GIT_NAME:-}" ]];  then git config user.name  "$GIT_NAME";  fi
if [[ -n "${GIT_EMAIL:-}" ]]; then git config user.email "$GIT_EMAIL"; fi

# ---- 2. 올라갈 파일 확인 ------------------------------------------------------
git add -A
say "올라갈 파일: $(git diff --cached --name-only | wc -l)개 (변경분 기준), 저장소 전체 $(git ls-files | wc -l)개"
total=$(git ls-files -z | xargs -0 du -cb 2>/dev/null | tail -1 | cut -f1)
say "올라갈 파일 합계: $(numfmt --to=iec "$total" 2>/dev/null || echo "$total B")"
big=$(git ls-files -z | xargs -0 -I{} find {} -maxdepth 0 -size +50M 2>/dev/null || true)
[[ -n "$big" ]] && fail "50 MB 넘는 파일이 있습니다 (GitHub 경고/거부 대상): $big"
if [[ $DRY_RUN -eq 1 ]]; then
    git ls-files | sed 's/^/    /'
    if git rev-parse -q --verify HEAD >/dev/null; then git reset -q; else git rm -r -q --cached . >/dev/null; fi
    say "--dry-run: 커밋/푸시하지 않았습니다."
    exit 0
fi

# ---- 3. 커밋 ---------------------------------------------------------------
author="$(git config user.name || true) <$(git config user.email || true)>"
say "커밋 작성자: $author  (바꾸려면 GIT_NAME / GIT_EMAIL)"
if git diff --cached --quiet; then
    say "새로 커밋할 변경이 없습니다."
else
    git commit -q -m "$COMMIT_MSG"
    say "커밋: $(git log -1 --oneline)"
fi

# ---- 4. GitHub 저장소 만들기 (없으면) -----------------------------------------
api="https://api.github.com"
if [[ "$USE_SSH" != "1" ]]; then
    [[ -n "${GITHUB_TOKEN:-}" ]] || fail "GITHUB_TOKEN 이 필요합니다. 예) GITHUB_TOKEN=ghp_xxx ./upload_to_github.sh"
    code=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $GITHUB_TOKEN" \
                "$api/repos/$GITHUB_ORG/$REPO_NAME")
    if [[ "$code" == "404" ]]; then
        say "GitHub 에 $GITHUB_ORG/$REPO_NAME 저장소를 만듭니다 ($VISIBILITY)"
        private=true; [[ "$VISIBILITY" == "public" ]] && private=false
        body="{\"name\":\"$REPO_NAME\",\"private\":$private,\"description\":\"side_plate 빨간 홈/문양 방향 인식 (STL 맞추기 + Photoneo 2D)\"}"
        resp=$(curl -s -w '\n%{http_code}' -H "Authorization: Bearer $GITHUB_TOKEN" \
                    -H "Accept: application/vnd.github+json" -d "$body" "$api/orgs/$GITHUB_ORG/repos")
        rc=$(tail -1 <<<"$resp")
        if [[ "$rc" == "404" ]]; then          # 조직이 아니라 개인 계정이면
            resp=$(curl -s -w '\n%{http_code}' -H "Authorization: Bearer $GITHUB_TOKEN" \
                        -H "Accept: application/vnd.github+json" -d "$body" "$api/user/repos")
            rc=$(tail -1 <<<"$resp")
        fi
        [[ "$rc" == "201" ]] || fail "저장소를 만들지 못했습니다 (HTTP $rc): $(head -n -1 <<<"$resp" | head -c 300)"
    elif [[ "$code" == "200" ]]; then
        say "GitHub 에 $GITHUB_ORG/$REPO_NAME 저장소가 이미 있습니다. 그대로 푸시합니다."
    else
        fail "저장소 확인 실패 (HTTP $code). 토큰 권한과 조직 이름($GITHUB_ORG)을 확인하세요."
    fi
fi

# ---- 5. 푸시 ---------------------------------------------------------------
clean_url="https://github.com/$GITHUB_ORG/$REPO_NAME.git"
[[ "$USE_SSH" == "1" ]] && clean_url="git@github.com:$GITHUB_ORG/$REPO_NAME.git"
if git remote get-url origin >/dev/null 2>&1; then
    git remote set-url origin "$clean_url"
else
    git remote add origin "$clean_url"
fi
say "푸시: $clean_url ($BRANCH)"
if [[ "$USE_SSH" == "1" ]]; then
    git push -u origin "$BRANCH"
else
    # 토큰은 이 명령에서만 쓰고 .git/config 에는 남기지 않는다
    auth=$(printf 'x-access-token:%s' "$GITHUB_TOKEN" | base64 -w0)
    git -c http.extraHeader="Authorization: Basic $auth" push -u origin "$BRANCH"
fi
say "완료: https://github.com/$GITHUB_ORG/$REPO_NAME"
