import re

def extract_answer(text):
    # Regex pattern to find numbers preceded by "answer" within the last 50 characters
    pattern = r'(\$\d+\$).*?answer|answer.*?(\$\d+\$)'
    
    # Find all matches in the text
    matches = re.findall(pattern, text.lower()[-100:])  # Only search in the last 50 characters
    
    if matches:
        # Extract just the numbers, stripping the dollar signs
        answers = [match[0].strip('$') if match[0] else match[1].strip('$') for match in matches]
        return int(answers[-1])
    else:
        return "No answer found"

# Example usage:
texts = [
    "answer $22$ herefore, $f(100)=15$. The answer is: $15$",
    "herefore, $f(100)=15$. The answer is $15$",
    "herefore, $f(100)=15$.  answer      $15$",
    "heslkdjfbsb answer will be $15$",
     "We calculated the result earlier, and the answer is $42$",  # Correct answer near the end
    "The computation was complex, but ultimately, the answer was $100$.",  # Answer with different number
    "This is a longer document that spans multiple paragraphs. In conclusion, the answer is $78$.",  # Longer distance to "answer"
    "Here's a tricky one: should have an answer $1234$ in it, but it's too far from the word answer",  # Answer is too far from the keyword
    "Answer to the universe is $42$, which is correctly placed near the end.",  # Two instances of the keyword "answer"
    "No answer in this one.",  # No answer provided
    "Edge case with answer right at the end $999$",  # Answer at the very end
    "Almost at the end answer $12$",  # Close to the edge case
    "What is the answer $5$?",  # Question format
    "The final answer, as computed, is very close to zero, which is $101000$.",  # Answer with zero,
    "The final, as computed, is very close to zero, which is $101000$, which is the answer."
    "The answer is $99$ but $0$ is not the answer$"
]

for text in texts:
    print(extract_answer(text))


