import os
from together import Together
import dotenv
import pandas as pd
from openai import OpenAI
dotenv.load_dotenv()
import time
from tqdm import tqdm

def naive_parse(answer):
    out = []
    start = False
    end = False
    for l in reversed(list(answer)):
        if l in '0123456789' and not end:
            start = True
            out.append(l)
        else:
            if start:
                end = True
        
    out = reversed(out)
    return ''.join(out)

client_Together = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
client_openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

train_set = pd.read_csv("train.csv")
try:
    benchmark_results_df = pd.read_csv("benchmark_results_v2.csv")
except:
    benchmark_results_df = train_set.copy()


models_to_benchmark = ["google/gemma-7b-it", 
                       'meta-llama/Llama-3-8b-hf',
                       "meta-llama/Llama-3-70b-chat-hf", 
                       'gpt-4-turbo']

system_prompt_keras = """Role:\nYou are an advanced AI system with exceptional mathematical reasoning and problem-solving capabilities, specifically designed to solve tricky math problems (whose answer is a non-negative integer) written in LaTeX format from the AI Mathematical Olympiad (AIMO) competition. Your task is to accurately analyze and solve intricate mathematical problems, demonstrating a deep understanding of mathematical concepts and a strong ability to apply logical reasoning strategies.\n\nInstruction:
1. Carefully read and comprehend the problem statement provided in the "Problem" section.
2. In the "Solution" section, provide a solution of the problem with detailed explanation of your logical reasoning process. Keep in mind that answer must be a non-negative integer number.
3. At the end, create a "Answer" section where you will state only the final integer answer, without any additional text or narrative. \n\nProblem:\n...\n\nSolution:\n...\n\nAnswer:\n..."""
system_prompt_naive = """You are an advanced AI system with exceptional mathematical reasoning and problem-solving capabilities, specifically designed to solve tricky math problems from AIME Math Olympiad."""

# add techniques and strategies for different types of 
prefix_before_problem = """
The AIME tests mathematical problem solving with arithmetic, algebra, counting, geometry, number theory, and probability and other secondary school math topics. Problems usually require either very creative use of secondary school curriculum, or an understanding as to how different areas of math can be used together to investigate and solve a problem. Recommended reading

Problem and solution books for past AMC exams. One of these books also includes numerous past AIMEs and solutions.
Introduction to Counting & Probability by Dr. David Patrick is recommended for students who qualify for the AIME, but feel they lag behind in their understanding of basic combinatorics and probability relative to their other areas of math. Information
Introduction to Geometry by Richard Rusczyk. Information
The Art of Problem Solving Volume II by Sandor Lehoczky and Richard Rusczyk. Information.

Problem:
{problem}

Now, first, identify the type of problem, list down some strategies to try, and try those strategies untill you solve the problem. The final answer should be a non-negative integer from 0 to 999. Clearly state the final answer at the end.
"""

for model in models_to_benchmark:
    if model not in benchmark_results_df.columns:
        df_analysis = train_set.copy()
        answers_parsed = []
        answer_full = []
        if model == 'gpt-4-turbo':
            client = client_openai
        else:
            client = client_Together
        for i, problem in tqdm(train_set.iterrows()):
            solution_answer = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content":  system_prompt_naive},
                        {"role": "user", "content": f"{problem['problem']}"}],
                #temperature=0.9,
            ).choices[0].message.content
            answer_full.append(solution_answer)
            try:
                answers_parsed.append(naive_parse(solution_answer))
            except:
                answers_parsed.append("Error")
        benchmark_results_df[model] = answers_parsed
        print(f"Model {model} benchmarked successfully")
        df_analysis["answer_full"] = answer_full
        df_analysis['answers_parsed'] = answers_parsed
        df_analysis.to_csv(f"10_train_{model[model.find('/')+1:]}.csv", index=False)
    else:
        print(f"Model {model} already benchmarked")

benchmark_results_df.to_csv("benchmark_results_v2.csv", index=False)