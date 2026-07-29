#!/usr/bin/env bash

# 检测是否被 sh 调用（dash 不支持 pipefail）
if [ -z "${BASH_VERSION:-}" ]; then
    echo -e "\033[0;31m[✗] 此脚本需要 bash，请用以下方式运行:\033[0m"
    echo "    bash $0"
    exit 1
fi
set -euo pipefail

# ════════════════════════════════════════════════════════════════
# Video2Mimic 全流程交互式脚本
# 视频裁剪 → GVHMR 人体姿态提取 → GMR 机器人重定向 → CSV导出
#
# 用法:
#   chmod +x run_pipeline.sh
#   ./run_pipeline.sh
#
# 跳过已完成的步骤: 每步检测输出文件是否存在，可选跳过/覆盖/退出
# ════════════════════════════════════════════════════════════════

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${ROOT_DIR}/run_pipeline_${TIMESTAMP}.log"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

log()  { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[✗]${NC} $*"; }
info() { echo -e "${CYAN}[i]${NC} $*"; }
sep()  { echo -e "${CYAN}──────────────────────────────────────────────${NC}"; }

# ── 检查已存在的文件，询问跳过/覆盖/退出 ──
# 用法: check_existing <文件路径> [描述]
# 返回: 0 = 跳过，1 = 覆盖/继续
check_existing() {
    local path="$1"
    local desc="${2:-文件}"
    if [ -f "$path" ] || [ -d "$path" ]; then
        warn "$desc 已存在: $path"
        echo "  [s] 跳过此步骤（使用现有结果）"
        echo "  [o] 覆盖（重新运行）"
        echo "  [q] 退出脚本"
        read -r -p "选择 [s/o/q] (默认 s): " choice
        case "${choice:-s}" in
            o|O)   return 1 ;;
            q|Q)   err "用户退出"; exit 0 ;;
            *)     log "跳过" ; return 0 ;;
        esac
    fi
    return 1
}

# ── 检查 conda ──
check_conda() {
    if ! command -v conda &> /dev/null; then
        err "conda 未找到，请先安装 miniconda/anaconda"
        exit 1
    fi
}

# ── 检查文件存在 ──
check_file() {
    if [ ! -f "$1" ]; then
        err "文件不存在: $1"
        return 1
    fi
}

# ── 检查 conda 环境 ──
check_env() {
    local env_name="$1"
    if ! conda env list | grep -q "^${env_name}\s"; then
        warn "conda 环境 '${env_name}' 不存在"
        info "请先安装: conda create -n ${env_name} python=3.10 -y"
        read -r -p "是否继续跳过检查？ [y/N] " skip
        if [[ ! "$skip" =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# ── 初始化 conda（后续可直接 conda activate） ──
init_conda() {
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
}


# ════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════
main() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║        Video2Mimic 全流程交互脚本           ║${NC}"
    echo -e "${CYAN}║  视频 → 人体姿态 → 机器人重定向 → CSV导出  ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════╝${NC}"
    echo ""

    check_conda
    init_conda

    # ════════════════════════════════════════════════════════════
    # 输入视频
    # ════════════════════════════════════════════════════════════
    sep
    echo -e "${CYAN}Step 0: 输入视频${NC}"
    sep

    while true; do
        read -r -p "请输入视频路径: " INPUT_VIDEO
        INPUT_VIDEO="$(eval echo "$INPUT_VIDEO")"
        [ -f "$INPUT_VIDEO" ] && break
        err "文件不存在，请重新输入"
    done
    VIDEO_NAME="$(basename "$INPUT_VIDEO")"
    VIDEO_NAME="${VIDEO_NAME%.*}"
    log "输入视频: $INPUT_VIDEO"

    # ════════════════════════════════════════════════════════════
    # Step 0a: 视频裁剪
    # ════════════════════════════════════════════════════════════
    sep
    echo -e "${CYAN}Step 0a: 视频裁剪（可选）${NC}"
    sep

    VIDEO_FOR_GVHMR="$INPUT_VIDEO"
    read -r -p "是否裁剪视频？ [y/N] " DO_TRIM
    if [[ "$DO_TRIM" =~ ^[Yy]$ ]]; then
        read -r -p "  裁掉开头秒数 [默认 0]: " TRIM_START
        TRIM_START="${TRIM_START:-0}"
        read -r -p "  裁掉结尾秒数 [默认 0]: " TRIM_END
        TRIM_END="${TRIM_END:-0}"
        TRIMMED_VIDEO="${ROOT_DIR}/trimmed_${VIDEO_NAME}.mp4"

        if check_existing "$TRIMMED_VIDEO" "裁剪结果"; then
            :  # 跳过
        else
            log "裁剪视频: 去头 ${TRIM_START}s, 去尾 ${TRIM_END}s"
            python "${ROOT_DIR}/video_editing.py" "$INPUT_VIDEO" \
                -s "$TRIM_START" -e "$TRIM_END" -o "$TRIMMED_VIDEO"
            log "裁剪完成"
        fi
        VIDEO_FOR_GVHMR="$TRIMMED_VIDEO"
    fi

    # ════════════════════════════════════════════════════════════
    # Step 0b: 音频融合
    # ════════════════════════════════════════════════════════════
    read -r -p "是否添加音频？ [y/N] " DO_AUDIO
    if [[ "$DO_AUDIO" =~ ^[Yy]$ ]]; then
        read -r -p "  音频文件路径: " AUDIO_FILE
        read -r -p "  音频延迟秒数 [默认 0]: " AUDIO_START
        AUDIO_START="${AUDIO_START:-0}"
        WITH_AUDIO="${ROOT_DIR}/${VIDEO_NAME}_with_audio.mp4"

        check_file "$AUDIO_FILE" || exit 1

        if check_existing "$WITH_AUDIO" "音视频融合结果"; then
            :  # 跳过
        else
            log "融合音频: ${AUDIO_FILE} (延迟 ${AUDIO_START}s)"
            python "${ROOT_DIR}/audio_video_merge.py" "$VIDEO_FOR_GVHMR" "$AUDIO_FILE" \
                -s "$AUDIO_START" -o "$WITH_AUDIO"
            log "音频融合完成"
        fi
        VIDEO_FOR_GVHMR="$WITH_AUDIO"
    fi

    # ════════════════════════════════════════════════════════════
    # Step 1: GVHMR — 视频 → SMPLX
    # ════════════════════════════════════════════════════════════
    sep
    echo -e "${CYAN}Step 1: GVHMR — 视频 → 人体SMPLX姿态${NC}"
    sep

    check_env "gvhmr"

    GVHMR_DIR="${ROOT_DIR}/GVHMR"
    GVHMR_OUTPUT_DIR="${GVHMR_DIR}/outputs/demo/${VIDEO_NAME}"
    SMPLX_PT="${GVHMR_OUTPUT_DIR}/hmr4d_results.pt"

    read -r -p "是否静态相机（跳过视觉里程计）？ [Y/n] " STATIC_CAM
    GVHMR_EXTRA=""
    if [[ ! "$STATIC_CAM" =~ ^[Nn]$ ]]; then
        GVHMR_EXTRA="-s"
        info "使用 -s (静态相机)"
    fi

    if check_existing "$SMPLX_PT" "GVHMR 结果"; then
        log "使用已有 GVHMR 结果"
    else
        conda activate gvhmr
        cd "$GVHMR_DIR"
        set +e
        python tools/demo/demo.py --video="$VIDEO_FOR_GVHMR" $GVHMR_EXTRA
        GVHMR_EXIT=$?
        set -euo pipefail
        cd "$ROOT_DIR"

        if [ $GVHMR_EXIT -ne 0 ]; then
            err "GVHMR 运行失败，请检查日志。可尝试使用绝对路径的视频。"
            exit 1
        fi

        # 如果输出路径与预期不符，尝试 glob 查找
        if [ ! -f "$SMPLX_PT" ]; then
            FOUND=$(find "${GVHMR_DIR}/outputs/demo/" -name "*.pt" -maxdepth 3 2>/dev/null | head -1)
            if [ -n "$FOUND" ]; then
                SMPLX_PT="$FOUND"
                warn "SMPLX 结果路径与预期不同，自动发现: $SMPLX_PT"
            else
                err "未找到 GVHMR 输出的 .pt 文件"
                info "请手动检查 GVHMR/outputs/demo/ 目录"
                exit 1
            fi
        fi
        log "GVHMR 完成！"
    fi
    log "SMPLX 输入: $SMPLX_PT"

    # ════════════════════════════════════════════════════════════
    # Step 2: GMR — SMPLX → 机器人DOF
    # ════════════════════════════════════════════════════════════
    sep
    echo -e "${CYAN}Step 2: GMR — 人体姿态 → 机器人重定向${NC}"
    sep

    check_env "gmr"

    # 机器人选择
    echo "支持的机器人类型:"
    echo "  1) unitree_g1            (29 DOF, 默认)"
    echo "  2) unitree_g1_23dof      (23 DOF)"
    echo "  3) unitree_g1_with_hands (43 DOF, 含灵巧手)"
    read -r -p "选择 [1/2/3] (默认 1): " ROBOT_CHOICE
    case "${ROBOT_CHOICE:-1}" in
        1) ROBOT="unitree_g1" ;;
        2) ROBOT="unitree_g1_23dof" ;;
        3) ROBOT="unitree_g1_with_hands" ;;
        *) ROBOT="unitree_g1" ;;
    esac
    log "机器人: $ROBOT"

    OUTPUT_DIR="${ROOT_DIR}/GMR/motions/${ROBOT}"
    mkdir -p "$OUTPUT_DIR"
    OUTPUT_PKL="${OUTPUT_DIR}/${VIDEO_NAME}.pkl"

    # 接地修正
    read -r -p "启用脚部接地修正？ [Y/n] " DO_GROUND
    GROUND_FLAG=""; GROUND_MODE=""
    if [[ ! "$DO_GROUND" =~ ^[Nn]$ ]]; then
        GROUND_FLAG="--ground_contact"
        read -r -p "  接地修正模式 [per_frame/global] (默认 per_frame): " GROUND_MODE
        GROUND_MODE="${GROUND_MODE:-per_frame}"
    fi

    # 默认姿态填充
    read -r -p "启用首尾默认姿态填充？ [Y/n] " DO_PAD
    PAD_FLAGS=""
    if [[ ! "$DO_PAD" =~ ^[Nn]$ ]]; then
        PAD_FLAGS="--pad_default_pose"
        read -r -p "  过渡帧数 [默认 15]: " PAD_FRAMES
        PAD_FRAMES="${PAD_FRAMES:-15}"
        PAD_FLAGS="$PAD_FLAGS --pad_frames $PAD_FRAMES"

        read -r -p "  保持帧数 [默认 30]: " HOLD_FRAMES
        HOLD_FRAMES="${HOLD_FRAMES:-30}"
        PAD_FLAGS="$PAD_FLAGS --pad_hold_frames $HOLD_FRAMES"

        read -r -p "  起始面朝角度（度, 绕Z轴）[默认 90]: " FACE_START
        FACE_START="${FACE_START:-90}"
        read -r -p "  结束面朝角度（度, 绕Z轴）[默认 0]: " FACE_END
        FACE_END="${FACE_END:-0}"
        PAD_FLAGS="$PAD_FLAGS --default_facing_angle $FACE_START $FACE_END"
    fi

    read -r -p "录制可视化视频？ [Y/n] " DO_VIDEO
    VIDEO_FLAG=""
    if [[ ! "$DO_VIDEO" =~ ^[Nn]$ ]]; then
        VIDEO_FLAG="--record_video"
    fi

    read -r -p "按原始帧率播放（--rate_limit）？ [y/N] " DO_RATE
    RATE_FLAG=""
    if [[ "$DO_RATE" =~ ^[Yy]$ ]]; then
        RATE_FLAG="--rate_limit"
    fi

    # 参数汇总
    echo ""
    info "重定向参数汇总:"
    echo "  GVHMR 输入 : $SMPLX_PT"
    echo "  机器人      : $ROBOT"
    echo "  接地修正    : ${GROUND_FLAG:-否}"
    [[ -n "$GROUND_MODE" ]] && echo "  接地模式    : $GROUND_MODE"
    echo "  默认姿态    : ${PAD_FLAGS:-否}"
    echo "  录制视频    : ${VIDEO_FLAG:-否}"
    echo "  限帧率      : ${RATE_FLAG:-否}"
    echo "  输出 PKL   : $OUTPUT_PKL"
    echo ""

    if check_existing "$OUTPUT_PKL" "机器人重定向结果"; then
        log "使用已有重定向结果"
    else
        conda activate gmr
        cd "${ROOT_DIR}/GMR"

        set +e
        python scripts/gvhmr2robot_fix.py \
            --gvhmr_pred_file "$SMPLX_PT" \
            --robot "$ROBOT" \
            $GROUND_FLAG \
            ${GROUND_FLAG:+--ground_contact_mode "${GROUND_MODE:-per_frame}"} \
            $PAD_FLAGS \
            $VIDEO_FLAG \
            $RATE_FLAG \
            --save_path "$OUTPUT_PKL"
        GMR_EXIT=$?
        set -euo pipefail

        cd "$ROOT_DIR"

        if [ $GMR_EXIT -ne 0 ]; then
            err "GMR 重定向失败"
            exit 1
        fi

        log "重定向完成！"
    fi
    log "输出 PKL: $OUTPUT_PKL"

    # ════════════════════════════════════════════════════════════
    # Step 3: PKL → CSV
    # ════════════════════════════════════════════════════════════
    sep
    echo -e "${CYAN}Step 3: PKL → CSV 导出${NC}"
    sep

    CSV_DIR="GMR/motions/${ROBOT}/csv"
    CSV_EXAMPLE="${ROOT_DIR}/${CSV_DIR}/${VIDEO_NAME}.csv"

    read -r -p "导出 CSV（供 BeyondMimic/RL 训练）？ [Y/n] " DO_CSV
    if [[ ! "$DO_CSV" =~ ^[Nn]$ ]]; then
        if check_existing "$CSV_EXAMPLE" "CSV 结果"; then
            log "使用已有 CSV"
        else
            conda activate gmr 2>/dev/null || true
            cd "${ROOT_DIR}/GMR"
            python scripts/batch_gmr_pkl_to_csv.py --folder "motions/${ROBOT}/"
            cd "$ROOT_DIR"
            log "CSV 导出完成"
        fi
        if [ -d "$CSV_DIR" ]; then
            ls -lh "${ROOT_DIR}/${CSV_DIR}/"*.csv 2>/dev/null || warn "CSV 目录为空"
        fi
    fi

    # ════════════════════════════════════════════════════════════
    # 完成
    # ════════════════════════════════════════════════════════════
    sep
    echo -e "${GREEN}╔══════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║              全流程完成！                    ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════╝${NC}"
    echo ""
    echo "  输出文件:"
    echo "    GVHMR SMPLX:  ${SMPLX_PT}"
    echo "    机器人PKL:    ${OUTPUT_PKL}"
    echo "    CSV目录:      ${ROOT_DIR}/${CSV_DIR}/"
    [ -f "${OUTPUT_PKL%.pkl}.mp4" ] && echo "    重定向视频:   ${OUTPUT_PKL%.pkl}.mp4"
    echo ""
    log "日志: $LOG_FILE"
}

main "$@" 2>&1 | tee "$LOG_FILE"
