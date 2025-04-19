from __future__ import annotations

import pandas as pd
import json
import re
import uuid

import os
from typing import List, Dict, Any
from huggingface_hub import InferenceClient
import json
import logging
from transformers import AutoTokenizer
import torch
import traceback


#the agents are gonna follow 2 approaches, the lawyer agents follow react and judge follows reflection
#firstly coming to the lawyer agent,
class LawyerAgent:


    def __init__(self,
                 name: str,
                 system_prompt: str,
                 model: str = "microsoft/Phi-3-mini-4k-instruct",
                 db=None):
        self.name = name
        self.role = name
        self.description = system_prompt

        self.logger = logging.getLogger(name)
        self.log_think = True
        self.system_prompt = system_prompt.strip()
        self.history: List[Dict[str, str]] = []      # list of {"role": ..., "content": ...}
        self.client = InferenceClient(
                        model="microsoft/Phi-3-mini-4k-instruct",
                        token="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"
                    ) # make sure this env‑var is set
        self.tokenizer = AutoTokenizer.from_pretrained(model) 
        self.db = db        
        
    # ---- helper for HF prompt formatting ----------
    def _format_prompt(self, user_msg: str) -> str:
        """
        Formats a full prompt that includes
        * system prompt
        * prior turns
        * new user message
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_msg})

        # HF text-generation endpoints expect a single string.

        prompt = ""
        for m in messages:
            prompt += f"<|{m['role']}|>\n{m['content']}\n"
        prompt += "<|assistant|>\n"
        return prompt

    # ---- produce a reply --------------------------
    def respond(self, user_msg: str, **gen_kwargs) -> str:
        prompt = self._format_prompt(user_msg)
        completion = self.client.text_generation(
            prompt,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            stream=False,
            **gen_kwargs
        )
        answer = completion.strip()
        # keep chat memory
        self.history.append({"role": "user", "content": user_msg})
        self.history.append({"role": "assistant", "content": answer})
        return answer
    
    def load_case_from_csv(self, case_id: str) -> Dict[str, Any]:
            try:
                df = pd.read_csv("cases.csv")
                case_data = df[df["id"] == case_id]
                if case_data.empty:
                    self.logger.warning(f"No case found with ID: {case_id}")
                    return {}
                return case_data.iloc[0].to_dict()
            except Exception as e:
                self.logger.error(f"Error loading case data: {e}")
                return {}


    # ---- think bitch ------------------------------
    def plan(self, history_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.log_think:
            self.logger.info(f"Agent ({self.role}) starting planning phase")
        history_context = self.prepare_history_context(history_list)
        plans = self._get_plan(history_context)
        if self.log_think:
            self.logger.info(f"Agent ({self.role}) generated plans: {plans}")
        queries = self._prepare_queries(plans, history_context)
        if self.log_think:
            self.logger.info(f"Agent ({self.role}) prepared queries: {queries}")
        return {"plans": plans, "queries": queries}

    def prepare_history_context(self, history: List[Dict[str, str]]) -> str:
        return "\n".join(f"<|{m['role']}|>\n{m['content']}\n" for m in history)

    def _get_plan(self, history_context: str) -> Dict[str, bool]:
        instruction = f"You are a {self.role}. {self.description}\n\n"
        prompt = (
            "Your Honor, I need to access the courtroom database to inform my next course of action."
            "Can you please direct me to the database where I can retrieve the necessary information to generate a plan for the case at hand?"
            "Upon accessing the database, I will provide a well-structured JSON file outlining the plans and queries required to move forward with the trial."
        )
        response = self._hf_generate(instruction, prompt + "\n\n" + history_context)
        return self._extract_plans(self.extract_response(response))

    def _prepare_queries(self, plans: Dict[str, bool], history_context: str) -> Dict[str, str]:
        queries = {}
        if plans.get("experience"):
            queries["experience"] = self._prepare_experience_query(history_context)
        if plans.get("case"):
            queries["case"] = self._prepare_case_query(history_context)
        if plans.get("legal"):
            queries["legal"] = self._prepare_legal_query(history_context)
        return queries

    def _prepare_experience_query(self, history_context: str) -> str:
        instruction = f"You are a {self.role}. {self.description}\n\n"
        prompt = (
            "Your Honor, I need to access the courtroom database to retrieve information about my experience."
            "Can you please direct me to the database where I can find details about my qualifications, training, and previous cases?"
            "Upon accessing the database, I will provide a well-structured JSON file outlining the queries required to gather the necessary information."
        )
        response = self._hf_generate(instruction, prompt + "\n\n" + history_context)
        return self.extract_response(response)
    def _prepare_case_query(self, history_context: str) -> str:
        instruction = f"You are a {self.role}. {self.description}\n\n"
        prompt = (
            "Your Honor, I need to access the courtroom database to retrieve information about the case at hand."
            "Can you please direct me to the database where I can find details about the facts, evidence, and legal issues involved in this case?"
            "Upon accessing the database, I will provide a well-structured JSON file outlining the queries required to gather the necessary information."
        )
        response = self._hf_generate(instruction, prompt + "\n\n" + history_context)
        return self.extract_response(response)

    def _prepare_legal_query(self, history_context: str) -> str:
        instruction = f"You are a {self.role}. {self.description}\n\n"
        prompt = (
            "Your Honor, I need to access the courtroom database to retrieve information about relevant legal precedents and statutes."
            "Can you please direct me to the database where I can find details about similar cases and applicable legal provisions?"
            "Upon accessing the database, I will provide a well-structured JSON file outlining the queries required to gather the necessary information."
        )
        response = self._hf_generate(instruction, prompt + "\n\n" + history_context)
        return self.extract_response(response)

    # ---- execute bitch ------------------------------
    def execute(self, queries: Dict[str, str]) -> Dict[str, Any]:
        results = {}
        for query_type, query in queries.items():
            results[query_type] = self._execute_query(query_type, query)
        return results

    def _prepare_context(self, plan: Dict[str, Any], history_list: List[Dict[str, str]]) -> str:
        context = ""
        queries = plan["queries"]

        if self.db:
            if "experience" in queries:
                exp = self.db.query_experience_metadatas(queries["experience"], n_results=3)
                context += f"\nRelevant Experience:\n{exp}\n"
            if "case" in queries:
                cs = self.db.query_case_metadatas(queries["case"], n_results=3)
                context += f"\nCase Precedents:\n{cs}\n"
            if "legal" in queries:
                law = self.db.query_legal(queries["legal"], n_results=3)
                context += f"\nLegal References:\n{law}\n"

        context += "\nConversation History:\n" + self.prepare_history_context(history_list)
        return context

    def speak(self, context: str, prompt: str) -> str:
        instruction = f"You are a {self.role}. {self.description}"
        full_prompt = f"{context}\n\n{prompt}"
        return self._hf_generate(instruction, full_prompt)

    # ---- this is for helpers --------------------------------------------

    def prepare_history_context(self, history: List[Dict[str, str]]) -> str:
        return "\n".join(f"<|{m['role']}|>\n{m['content']}" for m in history)

    def _hf_generate(self, instruction: str, prompt: str) -> str:
        full_prompt = f"<|system|>\n{instruction}\n<|user|>\n{prompt}\n<|assistant|>\n"
        max_total_tokens=4096
        max_new_tokens=512

        tokens = self.tokenizer(full_prompt, return_tensors="pt")["input_ids"][0]
        if len(tokens) + max_new_tokens > max_total_tokens:
            max_prompt_tokens = max_total_tokens - max_new_tokens
            tokens = tokens[-max_prompt_tokens:]  # keep only last tokens
            full_prompt = self.tokenizer.decode(tokens, skip_special_tokens=True)

        return self.client.text_generation(
            full_prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            stream=False
        ).strip()

    def extract_response(self, response: str) -> Any:
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            self.logger.warning(f"Failed to parse JSON: {response}")
            return {"experience": False, "case": False, "legal": False}

    def _extract_plans(self, response: dict) -> Dict[str, bool]:
        return {
            "experience": bool(response.get("experience")),
            "case": bool(response.get("case")),
            "legal": bool(response.get("legal"))
        }

    # ---- extra step for making it better framed ------------------------------

    def step(self, history_list: List[Dict[str, str]], prompt: str, case_id: str) -> str:
            plan_result = self.plan(history_list)

            # Load case from CSV
            case_context = self.load_case_from_csv(case_id)

            # Execute DB queries
            execution_results = self.execute(plan_result)

            # Prepare full context with case data
            context = self._prepare_context(plan_result, history_list, case_context)

            # Generate final answer
            response = self.speak(context, prompt)
            self.history.append({"role": "user", "content": prompt})
            self.history.append({"role": "assistant", "content": response})
            return response

class JudgeAgent:

    def __init__(self,
                 name: str,
                 system_prompt: str,
                 description: str = "",
                 model: str = "microsoft/Phi-3-mini-4k-instruct",
                 db=None):
        self.name = name
        self.role = name
        self.system_prompt = system_prompt.strip()
        self.description = description.strip()
        self.client = InferenceClient(
            model=model,
            token= "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"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.db = db
        self.logger = logging.getLogger(name)
        self.log_think = True

    # ---- Load Case CSV ----
    def load_case_from_csv(self, case_id: str) -> Dict[str, Any]:
        try:
            df = pd.read_csv("cases.csv")
            case_data = df[df["id"] == case_id]
            if case_data.empty:
                self.logger.warning(f"No case found with ID: {case_id}")
                return {}
            return case_data.iloc[0].to_dict()
        except Exception as e:
            self.logger.error(f"Error loading case data: {e}")
            return {}

    # ---- Context Preps ----

    def prepare_history_context(self, history: List[Dict[str, str]]) -> str:
        return "\n\n".join(f"{entry['role']} ({entry['name']}):\n  {entry['content']}" for entry in history)

    def prepare_case_content(self, case_dict: Dict[str, Any], history_context: str) -> str:
        case_info = "\n".join(f"{k}: {v}" for k, v in case_dict.items()) if case_dict else "No structured case data available."
        instruction = "You are a judge. Summarize the following case in 3 sentences."
        full_case = f"Case Details:\n{case_info}\n\nConversation History:\n{history_context}"
        return self._hf_generate(instruction, full_case)

    def _hf_generate(self, instruction: str, prompt: str) -> str:
        full_prompt = f"<|system|>\n{instruction}\n<|user|>\n{prompt}\n<|assistant|>\n"

        max_total_tokens = 4096
        max_new_tokens = 512

        input_ids = self.tokenizer(full_prompt, return_tensors="pt")["input_ids"][0]
        if len(input_ids) + max_new_tokens > max_total_tokens:
            max_prompt_tokens = max_total_tokens - max_new_tokens
            input_ids = input_ids[-max_prompt_tokens:]
            full_prompt = self.tokenizer.decode(input_ids, skip_special_tokens=True)

        return self.client.text_generation(full_prompt, max_new_tokens=max_new_tokens, temperature=0.7).strip()

    def _parse_json(self, response: str) -> Dict[str, Any]:
        try:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                return json.loads(match.group())
        except json.JSONDecodeError:
            self.logger.warning("Failed to parse JSON from reflection output.")
        return {}

    # ---- Reflection Logic ----

    def reflect(self, history_list: List[Dict[str, str]], case_id: str) -> Dict[str, Any]:
        history_context = self.prepare_history_context(history_list)
        case_dict = self.load_case_from_csv(case_id)
        case_content = self.prepare_case_content(case_dict, history_context)

        legal_reflection = self._reflect_on_legal_knowledge(history_context)
        experience_reflection = self._reflect_on_experience(case_content, history_context)
        case_reflection = self._reflect_on_case(case_content, history_context)

        return {
            "legal_reflection": legal_reflection,
            "experience_reflection": experience_reflection,
            "case_reflection": case_reflection
        }

    def _reflect_on_legal_knowledge(self, history_context: str) -> Dict[str, Any]:
        if not self._need_legal_reference(history_context):
            return {"needed_reference": False}

        query = self._prepare_legal_query(history_context)
        laws = self.db.query_legal(query, n_results=3) if self.db else []
        processed = [self._process_law(law) for law in laws]
        for law in processed:
            self.add_to_legal(str(uuid.uuid4()), law["content"], law["metadata"])
        return {"needed_reference": True, "query": query, "laws": processed}

    def _need_legal_reference(self, history_context: str) -> bool:
        instruction = f"You are a {self.role}. {self.description}"
        prompt = (
            "Review the court history. Would referencing specific laws improve reasoning?\n"
            "Respond with 'true' or 'false'.\n\n"
            f"{history_context}"
        )
        result = self._hf_generate(instruction, prompt).lower()
        return "true" in result

    def _reflect_on_experience(self, case_content: str, history_context: str) -> Dict[str, Any]:
        instruction = f"You are {self.role}. {self.description}"
        prompt = f""" Given the following case and history, summarize actionable courtroom experience.
                      Return as JSON with keys: context, content, focus_points, guidelines.

                      Case:\n{case_content}\n\nHistory:\n{history_context}
                  """
        response = self._hf_generate(instruction, prompt)
        print("LLM raw experience response:", response)
        summary = self._parse_json(response)
        entry = {
            "id": str(uuid.uuid4()),
            "content": summary.get("context", "[missing context]"),
            "metadata": {
                "context": summary.get("content", "[missing content]"),
                "focusPoints": summary.get("focus_points", ""),
                "guidelines": summary.get("guidelines", "")
            }
        }
        self.add_to_experience(entry["id"], entry["content"], entry["metadata"])
        return entry

    def _reflect_on_case(self, case_content: str, history_context: str) -> Dict[str, Any]:
        summary = self._generate_case_summary(case_content, history_context)
        entry = {
            "id": str(uuid.uuid4()),
            "content": summary.get("content", "Default content if missing"),
            "metadata": {
                "caseType": summary.get("case_type", "Unknown case type"),
                "keywords": summary.get("keywords", []),
                "quick_reaction_points": summary.get("quick_reaction_points", []),
                "response_directions": summary.get("response_directions", [])
            }
        }
        self.add_to_case(entry["id"], entry["content"], entry["metadata"])
        return entry

    # ---- Verdict --------------------------

    def deliberate(self, reflections: Dict[str, Any], history_list: List[Dict[str, str]], case_id: str) -> str:
        history_context = self.prepare_history_context(history_list)
        case_dict = self.load_case_from_csv(case_id)
        case_info = "\n".join(f"{k}: {v}" for k, v in case_dict.items()) if case_dict else "No case info."

        instruction = f"You are {self.role}. {self.description}"
        prompt = f""" You have reviewed the trial. Use the reflections, case file, and court history to make a final ruling.

                      Legal Reflection:\n{reflections['legal_reflection']}
                      Experience Reflection:\n{reflections['experience_reflection']}
                      Case Reflection:\n{reflections['case_reflection']}
                      Case File:\n{case_info}
                      Court History:\n{history_context}

                      Write a clear, fair, and reasoned verdict.
                      Format: JSON with key `verdict` and value either 0 (Against) or 1 (In Favor).
                  """
        result = self._hf_generate(instruction, prompt)
        print("Deliberation output:", result)
        parsed = self._parse_json(result)
        return parsed.get("verdict", "No verdict parsed.")

    # ---- Helpers --------------------------

    def add_to_legal(self, id: str, content: str, metadata: Dict[str, Any]):
        if self.db:
            self.db.add_to_legal(id, content, metadata)

    def add_to_case(self, id: str, content: str, metadata: Dict[str, Any]):
        if self.db:
            self.db.add_to_case(id, content, metadata)

    def add_to_experience(self, id: str, content: str, metadata: Dict[str, Any]):
        if self.db:
            self.db.add_to_experience(id, content, metadata)

    def _prepare_legal_query(self, history_context: str) -> str:
        instruction = f"You are a {self.role}. {self.description}"
        prompt = "Generate a keyword query to find applicable laws for the case.\n\n" + history_context
        result = self._hf_generate(instruction, prompt)
        return result.strip()

    def _generate_case_summary(self, case_content: str, history_context: str) -> Dict[str, Any]:
        instruction = f"You are {self.role}. {self.description}"
        prompt = f""" Summarize this case to support quick judicial decision-making.
                      Return as JSON with keys: content, case_type, keywords, quick_reaction_points, response_directions.

                      Case:\n{case_content}\n\nHistory:\n{history_context}
                  """
        return self._parse_json(self._hf_generate(instruction, prompt))

    def _process_law(self, law: dict) -> Dict[str, Any]:
        content = f"{law['lawsName']} {law['articleTag']} {law['articleContent']}"
        return {"content": content, "metadata": {"lawName": law["lawsName"], "articleTag": law["articleTag"]}}

class CourtroomDB:
    def __init__(self, dataframe: pd.DataFrame):
        """
        Initializes the database wrapper.
        Expects a DataFrame with 'lawsName', 'articleTag', and 'articleContent'.
        """
        required_columns = {"lawsName", "articleTag", "articleContent"}
        if not required_columns.issubset(dataframe.columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")
        self.df = dataframe.copy()

    def _search(self, query: str, n_results: int = 1):
        """
        Perform keyword search in combined law fields.
        """
        keyword = query.lower()
        mask = (
            self.df["lawsName"].str.lower().str.contains(keyword, na=False) |
            self.df["articleTag"].str.lower().str.contains(keyword, na=False) |
            self.df["articleContent"].str.lower().str.contains(keyword, na=False)
        )
        matches = self.df[mask]
        return matches.head(n_results).to_dict(orient="records")

    def query_legal(self, query: str, n_results: int = 1):
        """
        Returns legal statute entries matching the query.
        """
        return self._search(query, n_results)

    def query_case_metadatas(self, query: str, n_results: int = 1):
        """
        Placeholder for future: precedent case queries.
        Currently returns law matches.
        """
        return self._search(query, n_results)

    def query_experience_metadatas(self, query: str, n_results: int = 1):
        """
        Placeholder for future: experience logic queries.
        Currently returns law matches.
        """
        return self._search(query, n_results)

    def _append_to_db(self, category: str, id: str, content: str, metadata: Dict[str, Any]):
        entry = {
            "id": id,
            "lawsName": metadata.get("lawName", ""),
            "articleTag": metadata.get("articleTag", ""),
            "articleContent": content,
            "category": category,
            "metadata": metadata
        }
        self.df = pd.concat([self.df, pd.DataFrame([entry])], ignore_index=True)

    def add_to_legal(self, id: str, content: str, metadata: Dict[str, Any]):
        self._append_to_db("Legal", id, content, metadata)

    def add_to_case(self, id: str, content: str, metadata: Dict[str, Any]):
        self._append_to_db("Case", id, content, metadata)

    def add_to_experience(self, id: str, content: str, metadata: Dict[str, Any]):
        self._append_to_db("Experience", id, content, metadata)

# System prompts for each agent
DEFENSE_SYSTEM = """
You are **Alex Carter**, lead *defense counsel*.
Goals:
• Protect the constitutional rights of the defendant.
• Raise reasonable doubt by pointing out missing evidence or alternative explanations.
• Be respectful to the Court and to opposing counsel.
Style:
• Crisp, persuasive, grounded in precedent and facts provided.
• When citing precedent: give short case name + year (e.g., *Miranda v. Arizona* (1966)).
Ethics:
• Do not fabricate evidence; admit uncertainty when required.
"""

PROSECUTION_SYSTEM = """
You are **Jordan Blake**, *Assistant District Attorney* for the State.
Goals:
• Present the strongest good‑faith case against the accused.
• Lay out facts logically, citing exhibits or witness statements when available.
• Anticipate and rebut common defense arguments.
Style:
• Formal but plain English; persuasive, with confident tone.
Ethics:
• Duty is to justice, not merely to win. Concede points when ethically required.
"""

DEFENDANT_SYSTEM = """
You are **Julian St. Clair**, defendant in a criminal trial.
Goals:
• Assert your innocence and clear your name.
• Avoid self-incrimination and exercise your right to remain silent when necessary.
• Convey a sense of remorse and cooperation with the legal process.
Style:
• Confident, yet respectful and humble in tone.
• Avoid being confrontational or aggressive in your responses.
Ethics:
• Be truthful in your testimony, but do not volunteer information that may be harmful to your case.
• Do not make false accusations or shift blame onto others.
"""

PLAINTIFF_SYSTEM = """
You are **Eleanor Reed**, lead *plaintiff's counsel*.
Goals:
• Obtain fair compensation for the plaintiff's losses and damages.
• Establish a clear and convincing narrative of the defendant's liability.
• Demonstrate the significance of the plaintiff's harm and its impact on their life.
Style:
• Clear, concise, and empathetic, with a focus on storytelling and emotional appeal.
• Use vivid, descriptive language to paint a picture of the plaintiff's experience.
Ethics:
• Represent the plaintiff's interests zealously, but with honesty and integrity.
• Avoid making misleading or deceiving statements to the Court or opposing counsel.
• Disclose all relevant information and evidence, even if unfavorable to the plaintiff's case.
"""

JUDGE_SYSTEM = """
You are **Evelyn Thompson**, preside as *trial judge*.
Goals:
• Ensure a fair and impartial trial, upholding the principles of justice and the rule of law.
• Manage the courtroom efficiently, maintaining order and decorum.
• Render well-reasoned and legally sound decisions, supported by relevant statutes and case law.
Style:
• Clear, concise, and authoritative, with a focus on clarity and transparency.
• When citing precedent: provide brief explanations of the relevance and application to the case at hand.
Ethics:
• Remain impartial and detached, avoiding even the appearance of bias or prejudice.
• Uphold the highest standards of integrity, avoiding conflicts of interest and maintaining the confidentiality of sensitive information.
"""

def init_agents(db):
    # Initialize agents here
    defense = LawyerAgent("Defense", DEFENSE_SYSTEM, db=db)
    prosecution = LawyerAgent("Prosecution", PROSECUTION_SYSTEM, db=db)
    defendant = LawyerAgent("Defendant", DEFENDANT_SYSTEM, db=db)
    plaintiff = LawyerAgent("Plaintiff", PLAINTIFF_SYSTEM, db=db)
    judge = JudgeAgent("Judge", JUDGE_SYSTEM, db=db)
    
    return defense, prosecution, defendant, plaintiff, judge

def run_trial(plaintiff, prosecution, defense, defendant, judge, case_id: str, case_df: pd.DataFrame, rounds: int = 1):


    history = []
        # Load the case from the CSV
    case_row = case_df[case_df["id"] == case_id]

    if case_row.empty:
        raise ValueError(f"No case found with ID {case_id}")
    case_background = case_row.iloc[0]["text"]  # adjust column if needed
    past_cases = "\n".join(case_df[case_df["id"] != case_id]["text"].dropna().astype(str).tolist()[:5])
  # use top 5 others


    def log(role, name, content):
        history.append({"role": role, "name": name, "content": content.strip()[:50]})  # Trim long messages
        print(f"{role.upper()} ({name}):\n{content.strip()[:30]}\n")  # Truncated print

    def short_context(n=6):
        return judge.prepare_history_context(history[-n:])

    print("==== Opening Statements ====\n")
    opening_prompt = f"Case Details: {case_background}\n\nRelevant Past Cases: {past_cases[:30]}\n\nGive your opening statement briefly."

    for role, agent in [("plaintiff", plaintiff), ("prosecution", prosecution), ("defendant", defendant), ("defense", defense)]:
        plan = agent.plan(history)
        agent.execute(plan["queries"])
        response = agent.speak(short_context(), prompt=opening_prompt)
        log(role, agent.name, response)

    print("==== Arguments ====\n")
    for i in range(rounds):
        for role, agent in [("plaintiff", plaintiff), ("prosecution", prosecution), ("defendant", defendant), ("defense", defense)]:
            plan = agent.plan(history)
            agent.execute(plan["queries"])
            prompt = f"Based on this case: {case_background[:30]}\nAnd similar past rulings: {past_cases[:30]}\nState your strongest point concisely."
            response = agent.speak(short_context(), prompt)
            log(role, agent.name, response)

        if i == 0:
            print("==== Judge Interjects ====")
            objection_prompt = f"Here are recent arguments:\n{short_context()}\n\nAre there any objectionable or weak points based on past precedents: {past_cases[:800]}?"
            judge_comment = judge._hf_generate(judge.system_prompt, objection_prompt)
            log("judge", judge.name, judge_comment)

    print("==== Rebuttals ====\n")
    for role, agent in [("prosecution", prosecution), ("defense", defense)]:
        rebut_prompt = f"Based on your opponent's argument and prior similar case rulings ({past_cases[:30]}), briefly rebut their argument."
        response = agent.speak(short_context(), rebut_prompt)
        log(role, agent.name, response)

    print("==== Closing ====\n")
    for role, agent in [("plaintiff", plaintiff), ("prosecution", prosecution), ("defendant", defendant), ("defense", defense)]:
        closing_prompt = f"Conclude your case in under 3 lines. Remember the case: {case_background[:30]} and similar rulings."
        response = agent.speak(short_context(), closing_prompt)
        log(role, agent.name, response)



    print("==== Verdict ====\n")
    verdicts = []

    verdicts = [{"id": case_id, "verdict": final_verdict}]


    reflections = judge.reflect(history)
    final_verdict = judge.deliberate(reflections, history)
    log("judge", judge.name, final_verdict)

    verdict_df = pd.DataFrame(verdicts)
    verdict_df.to_csv("verdicts.csv", index=False)

    return {
    "case": case_background,
    "history": history,
    "reflections": reflections,
    "verdict": final_verdict
}


def batch_run_all_cases(cases_csv_path: str, output_csv_path: str = "final_verdicts.csv"):
    case_df = pd.read_csv(cases_csv_path)
    db = pd.read_csv(cases_csv_path)  # Reuse same file as DB
    defense, prosecution, defendant, plaintiff, judge = init_agents(db)

    results = []

    for case_id in case_df["id"]:
        print(f"\nRunning trial for case ID: {case_id}")
        try:
            verdict_data = run_trial(plaintiff, prosecution, defense, defendant, judge, case_id=case_id, case_df=case_df)
            results.append({
                "id": case_id,
                "verdict": verdict_data["verdict"]
            })
        except Exception as e:
            print(f"Error processing case {case_id}: {e}")
            traceback.print_exc()  # Add this line
            results.append({
                "id": case_id,
                "verdict": "error"
            })

    pd.DataFrame(results).to_csv(output_csv_path, index=False)
    print(f"\n✅ All cases processed. Output saved to: {output_csv_path}")

if __name__ == "__main__":
    batch_run_all_cases("cases.csv")
