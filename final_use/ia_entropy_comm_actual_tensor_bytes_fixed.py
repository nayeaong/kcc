import os
import json
import yaml
import copy
import time
import torch
from typing import List

from src.tasks.utils import (
    get_merged_embedding_for_cipher,
    get_merged_ids_mask_hs_for_sde,
)

from src.tasks.base_task import BaseTask


class IATask(BaseTask):
    def __init__(self, agents, dataset, args):
        super().__init__(agents, dataset, args)
        with open(args.prompt_file) as fin:
            name = "mmlu" if args.dataset.startswith("mmlu_") else args.dataset
            self.prompt_template = yaml.safe_load(fin)[name]

    def run(self, data):
        question = data["question"]
        ground_truth = data["answer"]
        passages = data["passages"]

        detail = {
            "question": question,
            "ground_truth": ground_truth,
            "passages": passages,
        }

        prompts = copy.deepcopy(self.prompt_template)
        for kk in prompts:
            prompts[kk] = prompts[kk].replace("{question}", question)

        # 원본과 동일: private knowledge 분배
        allocate_psg = [[] for _ in range(self.args.agent_cnt)]
        for idx, psg in enumerate(passages):
            allocate_psg[idx % self.args.agent_cnt].append(psg)

        for agent, allo_psg in zip(self.agents, allocate_psg):
            segments = ""
            for idx, psg in enumerate(allo_psg):
                segments += f"Document {idx+1}: {psg}\n"
            agent.role_prompt = prompts["role_prompt"].replace("{segments}", segments)
            agent.private_knowledge = segments

        run_time, use_rounds, comm_log, comm_cost = self.run_func(prompts)

        # evaluate
        for agent_idx, agent in enumerate(self.agents):
            det = self.dataset.evaluate(
                output=agent.final_output_text,
                test_id=data["test_id"],
            )

            for met in det:
                if met in self.dataset.eval_metrics:
                    try:
                        det[met] = det[met].item()
                    except:
                        pass

            det["history"] = agent.history

            # NL agent가 저장한 entropy 기록 (threshold 계산용)
            det["entropy_log"] = copy.deepcopy(getattr(agent, "entropy_log", []))
            det["last_entropy"] = getattr(agent, "last_entropy", None)

            detail[f"agent_{agent_idx}"] = det

        detail["run_time(s)"] = run_time
        detail["use_rounds"] = use_rounds
        detail["entropy_log"] = comm_log
        detail["comm_cost"] = comm_cost
        detail["comm_mode"] = getattr(self, "comm_mode", None)
        detail["adaptive_threshold"] = getattr(self, "adaptive_threshold", None)
        detail["adaptive_threshold_source"] = getattr(self, "adaptive_threshold_source", None)

        self._print_final_sample_log(data, detail, use_rounds, comm_log)
        return detail

    def run_func(self, prompts):
        raise NotImplementedError

    # =========================================================
    # log helper
    # =========================================================
    def _fmt_entropy_value(self, x):
        if x is None:
            return "None"
        return f"{float(x):.4f}"

    def _fmt_entropy_list(self, values):
        if values is None:
            return "[]"
        cleaned = []
        for v in values:
            if v is None:
                continue
            cleaned.append(f"'{float(v):.4f}'")
        return "[" + ", ".join(cleaned) + "]"

    def _collect_problem_mean_entropy_from_comm_log(self, comm_log):
        vals = []
        for log in comm_log:
            if "current_entropy" in log and log["current_entropy"] is not None:
                vals.append(float(log["current_entropy"]))
            elif "entropy" in log:
                e = log["entropy"]
                if isinstance(e, list):
                    vals.extend([float(v) for v in e if v is not None])
                elif e is not None:
                    vals.append(float(e))

        if not vals:
            return None
        return sum(vals) / len(vals)

    def _fmt_metric_value(self, x):
        x = float(x)
        if abs(x - round(x)) < 1e-8:
            return str(int(round(x)))
        return f"{x:.4f}"

    def _print_final_sample_log(self, data, detail, use_rounds, comm_log):
        metric_name = self._pick_metric_name([detail])
        sample_score = self._get_sample_score(detail, metric_name)
        problem_mean_entropy = self._collect_problem_mean_entropy_from_comm_log(comm_log)

        result_text = "정답" if sample_score >= 1.0 - 1e-8 else "오답"
        metric_label = metric_name.upper()

        print(
            f"[Final {metric_label}] "
            f"test_id={data['test_id']} | "
            f"{metric_label}={self._fmt_metric_value(sample_score)} ({result_text}) | "
            f"use_rounds={use_rounds} | "
            f"problem_mean_entropy={self._fmt_entropy_value(problem_mean_entropy)}"
        )

    # =========================================================
    # COMM_MODE
    # =========================================================
    def _get_comm_mode(self):
        """
        COMM_MODE:
        - all_sde  : 모든 통신을 SDE로
        - adaptive : threshold로 SDE/NL 혼용
        """
        comm_mode = os.environ.get("COMM_MODE", "all_sde").strip().lower()
        if comm_mode not in ["all_sde", "adaptive"]:
            raise ValueError(
                f"Invalid COMM_MODE={comm_mode}. "
                f"Use 'all_sde' or 'adaptive'."
            )
        return comm_mode

    # =========================================================
    # threshold helper
    # =========================================================
    def _pick_metric_name(self, details):
        if not details:
            return self.dataset.eval_metrics[0]

        for met in ["em", "accuracy", "acc"]:
            if "agent_0" in details[0] and met in details[0]["agent_0"]:
                return met

        return self.dataset.eval_metrics[0]

    def _get_sample_score(self, detail, metric_name):
        marked = []
        no_marked = []

        for agent_idx in range(self.args.agent_cnt):
            agent_det = detail[f"agent_{agent_idx}"]
            val = float(agent_det[metric_name])

            if agent_det.get("marked_answer", False):
                marked.append(val)
            else:
                no_marked.append(val)

        if len(marked) > 0:
            return sum(marked) / len(marked)
        if len(no_marked) > 0:
            return sum(no_marked) / len(no_marked)
        return 0.0

    def _collect_entropy_values_from_detail(self, detail):
        values = []

        for agent_idx in range(self.args.agent_cnt):
            agent_det = detail.get(f"agent_{agent_idx}", {})
            logs = agent_det.get("entropy_log", [])

            for log in logs:
                if "entropy" in log and log["entropy"] is not None:
                    values.append(float(log["entropy"]))

        return values

    def build_threshold_result(self, details):
        """
        NL-only 실행 결과에서
        - 정답 샘플 entropy 평균
        - 오답 샘플 entropy 평균
        둘 다 계산
        """
        if not details:
            return {}

        metric_name = self._pick_metric_name(details)

        em1_values = []
        em0_values = []
        em1_sample_count = 0
        em0_sample_count = 0

        for detail in details:
            score = self._get_sample_score(detail, metric_name)
            entropy_values = self._collect_entropy_values_from_detail(detail)

            if not entropy_values:
                continue

            if score >= 1.0 - 1e-8:
                em1_values.extend(entropy_values)
                em1_sample_count += 1
            else:
                em0_values.extend(entropy_values)
                em0_sample_count += 1

        em1_mean = sum(em1_values) / len(em1_values) if em1_values else None
        em0_mean = sum(em0_values) / len(em0_values) if em0_values else None

        return {
            "metric_name": metric_name,
            "em1_entropy_threshold": em1_mean,
            "em0_entropy_threshold": em0_mean,
            "em1_sample_count": em1_sample_count,
            "em0_sample_count": em0_sample_count,
            "em1_entropy_count": len(em1_values),
            "em0_entropy_count": len(em0_values),
        }

    def load_entropy_threshold_from_env(self):
        """
        adaptive 실행 시 ENV에서 직접 threshold를 읽는다.
        사용 예:
            os.environ["COMM_MODE"] = "adaptive"
            os.environ["ENTROPY_THRESHOLD"] = f"{tau1:.4f}"
        """
        env_threshold = os.environ.get("ENTROPY_THRESHOLD", "").strip()

        if not env_threshold:
            raise ValueError(
                "adaptive 모드에서는 ENTROPY_THRESHOLD 환경변수가 반드시 필요합니다.\n"
                "예: os.environ['ENTROPY_THRESHOLD'] = f'{tau1:.4f}'"
            )

        try:
            threshold = float(env_threshold)
        except ValueError:
            raise ValueError(
                f"ENTROPY_THRESHOLD={env_threshold} 를 float으로 변환할 수 없습니다."
            )

        threshold_result = {
            "metric_name": "env_manual_threshold",
            "em1_entropy_threshold": threshold,
            "em0_entropy_threshold": None,
            "em1_sample_count": None,
            "em0_sample_count": None,
            "em1_entropy_count": None,
            "em0_entropy_count": None,
        }

        return threshold, "ENV:ENTROPY_THRESHOLD", threshold_result

    # =========================================================
    # communication cost helper
    # =========================================================
    def _empty_comm_cost(self):
        return {
            "total_bytes": 0.0,
            "text_bytes": 0.0,
            "nl_text_bytes": 0.0,
            "sde_text_bytes": 0.0,
            "sde_bytes": 0.0,
            "token_count": 0.0,
            "comm_count": 0.0,
            "nl_count": 0.0,
            "sde_count": 0.0,
        }

    def _utf8_bytes(self, text):
        if text is None:
            return 0
        return len(text.encode("utf-8"))

    def _get_hidden_size(self, agent):
        if hasattr(agent, "model") and hasattr(agent.model, "config"):
            hs = getattr(agent.model.config, "hidden_size", None)
            if hs is not None:
                return int(hs)

        if hasattr(agent, "assistant_hs") and len(agent.assistant_hs) > 0:
            maybe_dict = agent.assistant_hs[-1]
            if isinstance(maybe_dict, dict) and len(maybe_dict) > 0:
                first_key = next(iter(maybe_dict))
                return int(maybe_dict[first_key].shape[-1])

        raise ValueError("hidden_size를 찾지 못했습니다.")

    def _add_nl_comm(self, comm_cost, text):
        """
        NL 통신:
        전송량 = utf-8 text bytes
        """
        text_bytes = float(self._utf8_bytes(text))

        comm_cost["text_bytes"] += text_bytes
        comm_cost["nl_text_bytes"] += text_bytes
        comm_cost["total_bytes"] += text_bytes
        comm_cost["comm_count"] += 1.0
        comm_cost["nl_count"] += 1.0

        return {
            "text_bytes": text_bytes,
            "sde_bytes": 0.0,
            "total_bytes": text_bytes,
            "token_count": 0.0,
        }

    def _calc_sde_tensor_bytes(self, input_hs):
        """
        실제 SDE tensor의 dtype/shape를 사용해 바이트 수를 계산한다.
        float32로 4바이트 고정하지 않고, 각 tensor의 element_size()를 사용한다.
        """
        total = 0.0

        if input_hs is None:
            return total

        for _, hs in input_hs.items():
            if hs is None:
                continue
            total += float(hs.numel() * hs.element_size())

        return total

    def _add_sde_comm(self, comm_cost, text, token_count, input_hs):
        """
        SDE 통신량:
        total_bytes = text_bytes + 실제 state delta tensor bytes
        """
        text_bytes = float(self._utf8_bytes(text))
        sde_bytes = self._calc_sde_tensor_bytes(input_hs)
        total_bytes = text_bytes + sde_bytes

        comm_cost["text_bytes"] += text_bytes
        comm_cost["sde_text_bytes"] += text_bytes
        comm_cost["sde_bytes"] += sde_bytes
        comm_cost["total_bytes"] += total_bytes
        comm_cost["token_count"] += float(token_count)
        comm_cost["comm_count"] += 1.0
        comm_cost["sde_count"] += 1.0

        return {
            "text_bytes": text_bytes,
            "sde_bytes": sde_bytes,
            "total_bytes": total_bytes,
            "token_count": float(token_count),
        }

    def _finalize_comm_cost(self, comm_cost):
        out = copy.deepcopy(comm_cost)

        if out["comm_count"] > 0:
            out["bytes_per_message"] = out["total_bytes"] / out["comm_count"]
            out["sde_rate"] = out["sde_count"] / out["comm_count"]
        else:
            out["bytes_per_message"] = 0.0
            out["sde_rate"] = 0.0

        return out

    def _aggregate_comm_cost(self, details):
        avg = self._empty_comm_cost()

        if not details:
            return self._finalize_comm_cost(avg)

        for d in details:
            c = d.get("comm_cost", {})
            for k in avg:
                avg[k] += float(c.get(k, 0.0))

        for k in avg:
            avg[k] /= len(details)

        return self._finalize_comm_cost(avg)

    def generate_result(self, details):
        res = {
            "run_time(s)": sum(d["run_time(s)"] for d in details) / len(details),
            "use_rounds": sum(d["use_rounds"] for d in details) / len(details),
        }

        agent_cnt = self.args.agent_cnt
        for agent_idx in range(agent_cnt):
            agent_res = {
                met: sum(float(d[f"agent_{agent_idx}"][met]) for d in details) / len(details)
                for met in self.dataset.eval_metrics
            }
            res[f"agent_{agent_idx}"] = agent_res

        if self.args.method == "single":
            average = {}
            for met in self.dataset.eval_metrics:
                average[met] = max(
                    [float(res[f"agent_{agent_idx}"][met]) for agent_idx in range(agent_cnt)]
                )
        else:
            average = {}
            for met in self.dataset.eval_metrics:
                val = []
                for det in details:
                    marked = []
                    no_marked = []
                    for agent_idx in range(agent_cnt):
                        if det[f"agent_{agent_idx}"]["marked_answer"]:
                            marked.append(det[f"agent_{agent_idx}"][met])
                        else:
                            no_marked.append(det[f"agent_{agent_idx}"][met])
                    if len(marked) > 0:
                        val.append(sum(marked) / len(marked))
                    else:
                        val.append(sum(no_marked) / len(no_marked))
                average[met] = sum(val) / len(val)

        res["average"] = average
        res["comm_cost"] = self._aggregate_comm_cost(details)
        res["comm_mode"] = getattr(self, "comm_mode", None)

        if self.args.method == "nl":
            res["threshold_result"] = self.build_threshold_result(details)

        if hasattr(self, "adaptive_threshold") and self.adaptive_threshold is not None:
            res["adaptive_threshold"] = float(self.adaptive_threshold)
            res["adaptive_threshold_source"] = getattr(self, "adaptive_threshold_source", None)

        if hasattr(self, "threshold_result"):
            res["threshold_result_used"] = self.threshold_result

        return res


class SingleIATask(IATask):
    def run_func(self, prompts):
        start_time = time.perf_counter()
        comm_log = []
        comm_cost = self._empty_comm_cost()

        for agent in self.agents:
            prompt = prompts["direct_prompt"].replace("{segments}", agent.private_knowledge)
            agent.role_prompt = None
            agent.init_history(first_user_prompt=prompt)
            output = agent.generate(agent.history_msgs)
            agent.final_output_text = output
            agent.history = agent.history_msgs

        end_time = time.perf_counter()
        return end_time - start_time, 1, comm_log, self._finalize_comm_cost(comm_cost)


class NlIATask(IATask):
    def run_func(self, prompts):
        start_time = time.perf_counter()
        agent_cnt = self.args.agent_cnt
        use_rounds = -1
        comm_log = []
        comm_cost = self._empty_comm_cost()

        # 1라운드: 각 agent가 독립 생성 -> 통신량 0
        for agent_idx, agent in enumerate(self.agents):
            agent.init_history(first_user_prompt=prompts["first_prompt"])
            text = agent.generate(agent.history_msgs)
            cur_entropy = getattr(agent, "last_entropy", None)

            # rd=1은 통신 선택이 아니라 entropy만 기록
            comm_log.append({
                "round": 1,
                "agent": agent_idx,
                "entropy": [float(cur_entropy)] if cur_entropy is not None else [],
                "current_entropy": float(cur_entropy) if cur_entropy is not None else None,
                "threshold": None,
                "method": None,
                "text_bytes": 0.0,
                "sde_bytes": 0.0,
                "total_bytes": 0.0,
                "token_count": 0.0,
            })

            print(
                f"[Entropy] rd=1 agent={agent_idx} "
                f"entropies={self._fmt_entropy_list([cur_entropy] if cur_entropy is not None else [])}"
            )

            if any([w in text for w in self.dataset.stop_words]):
                use_rounds = 1

        # 2라운드부터: 원본과 동일한 NL 통신 + 전송량 기록 + entropy 로그 출력
        for rd in range(self.args.rounds - 1):
            if use_rounds != -1:
                break

            for cur in range(agent_cnt):
                agent = self.agents[cur]
                all_other_resp = ""

                turn_text_bytes = 0.0
                turn_total_bytes = 0.0

                for other in range(agent_cnt):
                    if other != cur:
                        sender_agent = self.agents[other]
                        other_resp = sender_agent.assistant_output[rd]

                        msg_cost = self._add_nl_comm(comm_cost, other_resp)
                        turn_text_bytes += msg_cost["text_bytes"]
                        turn_total_bytes += msg_cost["total_bytes"]

                        all_other_resp += prompts["other_response_prompt"].replace(
                            "{other_response}", other_resp
                        )

                agent.history_msgs.append({
                    "role": "user",
                    "content": prompts["communication_prompt"].replace(
                        "{all_other_response}", all_other_resp
                    )
                })

                text = agent.generate(agent.history_msgs)
                cur_entropy = getattr(agent, "last_entropy", None)

                comm_log.append({
                    "round": rd + 2,
                    "agent": cur,
                    "entropy": [float(cur_entropy)] if cur_entropy is not None else [],
                    "current_entropy": float(cur_entropy) if cur_entropy is not None else None,
                    "threshold": None,
                    "method": "NL",
                    "text_bytes": turn_text_bytes,
                    "sde_bytes": 0.0,
                    "total_bytes": turn_total_bytes,
                    "token_count": 0.0,
                })

                print(
                    f"[Entropy] rd={rd+2} agent={cur} "
                    f"entropies={self._fmt_entropy_list([cur_entropy] if cur_entropy is not None else [])} "
                    f"threshold=None -> NL"
                )

                if any([w in text for w in self.dataset.stop_words]):
                    use_rounds = rd + 2

        for agent in self.agents:
            agent.final_output_text = agent.assistant_output[-1]

        end_time = time.perf_counter()
        for agent in self.agents:
            agent.history = agent.history_msgs

        return end_time - start_time, use_rounds, comm_log, self._finalize_comm_cost(comm_cost)


class CipherIATask(IATask):
    def run_func(self, prompts):
        start_time = time.perf_counter()
        agent_cnt = self.args.agent_cnt
        use_rounds = -1
        comm_log = []
        comm_cost = self._empty_comm_cost()

        # 1라운드: 원본과 동일, 독립 생성 -> 통신량 0
        for agent in self.agents:
            agent.init_history(first_user_prompt=prompts["first_prompt"])
            output_embs = agent.generate(agent.history_embs)
            output_text = agent.get_human_output(output_embs)
            if any([w in output_text for w in self.dataset.stop_words]):
                use_rounds = 1

        # 2라운드부터: 원본과 동일 + text bytes 기록
        for rd in range(self.args.rounds - 1):
            if use_rounds != -1:
                break

            for cur in range(agent_cnt):
                agent = self.agents[cur]
                merged_other_embs = []

                turn_text_bytes = 0.0
                turn_total_bytes = 0.0

                for other in range(agent_cnt):
                    if cur != other:
                        sender_agent = self.agents[other]
                        other_text = sender_agent.get_human_output(sender_agent.assistant_output[rd])

                        msg_cost = self._add_nl_comm(comm_cost, other_text)
                        turn_text_bytes += msg_cost["text_bytes"]
                        turn_total_bytes += msg_cost["total_bytes"]

                        merged_other_embs.append(
                            get_merged_embedding_for_cipher(
                                t2e_func=agent.text_to_embedding,
                                prompt_template=prompts["other_response_prompt"],
                                placeholder="{other_response}",
                                input_embs=sender_agent.assistant_output[rd]
                            )
                        )

                comm_log.append({
                    "round": rd + 2,
                    "agent": cur,
                    "method": "CIPHER",
                    "text_bytes": turn_text_bytes,
                    "sde_bytes": 0.0,
                    "total_bytes": turn_total_bytes,
                    "token_count": 0.0,
                })

                user_embs = get_merged_embedding_for_cipher(
                    t2e_func=agent.text_to_embedding,
                    prompt_template=prompts["communication_prompt"],
                    placeholder="{all_other_response}",
                    input_embs=torch.cat(merged_other_embs, dim=0),
                )

                agent.history_embs = torch.cat([
                    agent.history_embs,
                    agent.user_embs_fr,
                    user_embs,
                    agent.user_embs_ed,
                    agent.assistant_embs_fr,
                ], dim=0)

                output_embs = agent.generate(agent.history_embs)
                output_text = agent.get_human_output(output_embs)
                if any([w in output_text for w in self.dataset.stop_words]):
                    use_rounds = rd + 2

        for agent in self.agents:
            agent.final_output_text = agent.get_human_output(agent.assistant_output[-1])

        end_time = time.perf_counter()
        for agent in self.agents:
            agent.history = agent.get_human_output(agent.history_embs)

        return end_time - start_time, use_rounds, comm_log, self._finalize_comm_cost(comm_cost)


class SDEIATask(IATask):
    def run_func(self, prompts):
        start_time = time.perf_counter()
        agent_cnt = self.args.agent_cnt
        use_rounds = -1
        comm_log = []
        comm_cost = self._empty_comm_cost()

        # COMM_MODE로 all_sde / adaptive 구분
        comm_mode = self._get_comm_mode()
        self.comm_mode = comm_mode
        adaptive_mode = (comm_mode == "adaptive")

        entropy_threshold = None
        threshold_source = None
        threshold_result = None

        # adaptive면 ENV의 ENTROPY_THRESHOLD 사용
        if adaptive_mode:
            entropy_threshold, threshold_source, threshold_result = self.load_entropy_threshold_from_env()
            self.adaptive_threshold = entropy_threshold
            self.adaptive_threshold_source = threshold_source
            self.threshold_result = threshold_result

        # ── 통신 흐름 ────────────────────────────────────────────────
        # 1라운드:
        #   생성 → 엔트로피 확인 → SDE/NL 결정 → send_mode[agent] 저장
        #   (통신량 0, 다음 라운드에서 이 send_mode로 전달)
        #
        # 2라운드~:
        #   상대방의 send_mode로 수신 → 생성 → 엔트로피 확인
        #   → 다음 라운드 send_mode 결정 → send_mode[cur] 갱신
        #
        # "내가 불확실하면(엔트로피 ≥ threshold) 내 응답을 SDE로 보낸다"
        # ─────────────────────────────────────────────────────────────

        round_entropies = []
        send_mode = []  # 각 에이전트가 다음 라운드에 보낼 방식

        # ── 1라운드: 독립 생성 + send_mode 결정 ──────────────────
        for agent_idx, agent in enumerate(self.agents):
            agent.init_history(first_user_prompt=prompts["first_prompt"])
            output_ids, _, entropy = agent.generate(
                input_ids=agent.history_ids,
                if_edit=False,
                edit_layer_idx=self.args.edit_layer_idx,
            )
            entropy = float(entropy)
            round_entropies.append(entropy)

            # 생성 후 본인 엔트로피로 다음 라운드 전송 방식 결정
            if adaptive_mode:
                use_sde = entropy >= entropy_threshold
            else:
                use_sde = True
            send_mode.append(use_sde)
            method_label = "SDE" if use_sde else "NL"

            comm_log.append({
                "round":           1,
                "agent":           agent_idx,
                "entropy":         [entropy],
                "current_entropy": entropy,
                "threshold":       float(entropy_threshold) if entropy_threshold is not None else None,
                "method":          method_label,  # 다음 라운드에 보낼 방식
                "text_bytes":      0.0,
                "sde_bytes":       0.0,
                "total_bytes":     0.0,
                "token_count":     0.0,
            })

            print(
                f"[Entropy] rd=1 agent={agent_idx} "
                f"entropy={self._fmt_entropy_value(entropy)} "
                f"threshold={self._fmt_entropy_value(entropy_threshold)} "
                f"-> next_send={method_label}"
            )

            output_text = agent.get_human_output(output_ids)
            if any([w in output_text for w in self.dataset.stop_words]):
                use_rounds = 1

        # ── 2라운드~: 상대방 send_mode로 수신 → 생성 → 다음 send_mode 결정 ──
        for rd in range(self.args.rounds - 1):
            if use_rounds != -1:
                break

            for cur in range(agent_cnt):
                agent = self.agents[cur]

                turn_text_bytes = 0.0
                turn_sde_bytes  = 0.0
                turn_total_bytes = 0.0
                turn_token_count = 0.0

                # 상대방의 send_mode에 따라 수신 방식 결정
                # (agent_cnt=2 기준: 상대방은 1명)
                other_agents = [o for o in range(agent_cnt) if o != cur]
                # 상대방들 중 하나라도 SDE로 보내기로 했으면 SDE로 받음
                recv_sde = any(send_mode[other] for other in other_agents)

                if recv_sde:
                    # ── SDE 수신 ──────────────────────────────────
                    merged_input_ids = []
                    merged_mask = []
                    merged_hs = []

                    for other in other_agents:
                        sender_agent = self.agents[other]
                        resp_ids = sender_agent.assistant_ids[rd]
                        other_resp_text = sender_agent.get_human_output(resp_ids)

                        msg_cost = self._add_sde_comm(
                            comm_cost=comm_cost,
                            text=other_resp_text,
                            token_count=len(resp_ids),
                            input_hs=sender_agent.assistant_hs[rd],
                        )
                        turn_text_bytes  += msg_cost["text_bytes"]
                        turn_sde_bytes   += msg_cost["sde_bytes"]
                        turn_total_bytes += msg_cost["total_bytes"]
                        turn_token_count += msg_cost["token_count"]

                        ids, mask, hs = get_merged_ids_mask_hs_for_sde(
                            tokenizer=agent.tokenizer,
                            prompt_template=prompts["other_response_prompt"],
                            placeholder="{other_response}",
                            input_ids=resp_ids,
                            input_mask=torch.zeros(len(resp_ids), dtype=torch.bool),
                            input_hs=sender_agent.assistant_hs[rd],
                        )
                        merged_input_ids.append(ids)
                        merged_mask.append(mask)
                        merged_hs.append(hs)

                    all_resp_ids = []
                    for ids in merged_input_ids:
                        all_resp_ids += ids

                    user_input_ids, user_mask, user_hs = get_merged_ids_mask_hs_for_sde(
                        tokenizer=agent.tokenizer,
                        prompt_template=prompts["communication_prompt"],
                        placeholder="{all_other_response}",
                        input_ids=all_resp_ids,
                        input_mask=torch.cat(merged_mask, dim=0),
                        input_hs={
                            layer_idx: torch.cat(
                                [merged_hs[_][layer_idx] for _ in range(len(merged_hs))],
                                dim=1
                            )
                            for layer_idx in self.args.edit_layer_idx
                        },
                    )

                    history_mask = torch.cat([
                        torch.zeros(len(agent.history_ids) + len(agent.user_prompt_fr), dtype=torch.bool),
                        user_mask,
                        torch.zeros(len(agent.user_prompt_ed) + len(agent.assistant_prompt_fr), dtype=torch.bool),
                    ], dim=0)

                    history_hs = {}
                    for layer_idx, hs in user_hs.items():
                        history_hs[layer_idx] = torch.cat([
                            torch.zeros((1, len(agent.history_ids) + len(agent.user_prompt_fr), hs.shape[-1])),
                            hs,
                            torch.zeros((1, len(agent.user_prompt_ed) + len(agent.assistant_prompt_fr), hs.shape[-1])),
                        ], dim=1)

                    agent.history_ids = (
                        agent.history_ids
                        + agent.user_prompt_fr
                        + user_input_ids
                        + agent.user_prompt_ed
                        + agent.assistant_prompt_fr
                    )

                    output_ids, _, entropy = agent.generate(
                        input_ids=agent.history_ids,
                        if_edit=True,
                        edit_layer_idx=self.args.edit_layer_idx,
                        edit_mask=history_mask,
                        edit_tensor=history_hs,
                    )
                    recv_label = "SDE"

                else:
                    # ── NL 수신 ───────────────────────────────────
                    all_other_resp = ""
                    for other in other_agents:
                        sender_agent = self.agents[other]
                        other_resp_ids = sender_agent.assistant_ids[rd]
                        other_resp_text = sender_agent.get_human_output(other_resp_ids)

                        msg_cost = self._add_nl_comm(comm_cost, other_resp_text)
                        turn_text_bytes  += msg_cost["text_bytes"]
                        turn_total_bytes += msg_cost["total_bytes"]

                        all_other_resp += prompts["other_response_prompt"].replace(
                            "{other_response}", other_resp_text
                        )

                    user_input_ids_nl = agent.tokenizer.encode(
                        prompts["communication_prompt"].replace(
                            "{all_other_response}", all_other_resp
                        ),
                        add_special_tokens=False,
                    )

                    agent.history_ids = (
                        agent.history_ids
                        + agent.user_prompt_fr
                        + user_input_ids_nl
                        + agent.user_prompt_ed
                        + agent.assistant_prompt_fr
                    )

                    output_ids, _, entropy = agent.generate(
                        input_ids=agent.history_ids,
                        if_edit=False,
                        edit_layer_idx=self.args.edit_layer_idx,
                    )
                    recv_label = "NL"

                entropy = float(entropy)

                # 생성 후 본인 엔트로피로 다음 라운드 전송 방식 결정
                if adaptive_mode:
                    next_use_sde = entropy >= entropy_threshold
                else:
                    next_use_sde = True
                send_mode[cur] = next_use_sde
                next_label = "SDE" if next_use_sde else "NL"

                comm_log.append({
                    "round":           rd + 2,
                    "agent":           cur,
                    "entropy":         [entropy],      # 생성 후 본인 엔트로피
                    "current_entropy": entropy,
                    "threshold":       float(entropy_threshold) if entropy_threshold is not None else None,
                    "method":          recv_label,     # 이번 라운드 수신 방식
                    "next_send":       next_label,     # 다음 라운드 전송 방식
                    "text_bytes":      turn_text_bytes,
                    "sde_bytes":       turn_sde_bytes,
                    "total_bytes":     turn_total_bytes,
                    "token_count":     turn_token_count,
                })

                print(
                    f"[Entropy] rd={rd+2} agent={cur} "
                    f"recv={recv_label} "
                    f"entropy={self._fmt_entropy_value(entropy)} "
                    f"threshold={self._fmt_entropy_value(entropy_threshold)} "
                    f"-> next_send={next_label}"
                )

                round_entropies[cur] = entropy

                output_text = agent.get_human_output(output_ids)
                if any([w in output_text for w in self.dataset.stop_words]):
                    use_rounds = rd + 2

        for agent in self.agents:
            agent.final_output_text = agent.get_human_output(agent.assistant_ids[-1])

        end_time = time.perf_counter()
        for agent in self.agents:
            agent.history = agent.get_human_output(agent.history_ids)

        return end_time - start_time, use_rounds, comm_log, self._finalize_comm_cost(comm_cost)