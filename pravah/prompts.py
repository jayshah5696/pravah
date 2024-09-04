from jinja2 import Environment, FileSystemLoader
import re
def generate_prompt_template(query, context_list, extra_context=None):
    """
    Generates an expanded prompt template using Jinja2 for a RAG system.

    Parameters:
    - query: The query input from the user.
    - context_list: A list of dictionaries, each containing 'content' and 'url' keys.

    Returns:
    - A string containing the final prompt with inline citations and structured instructions.
    """
    env = Environment(loader=FileSystemLoader(''))
    template_string = """
    <role>You are an intelligent Search Assistant swiss army knife powered for information discovery and curiosity.
    you are responsible for accurately address the given context and providing a concise response to the query.
    </role>

    <instructions>
        Think step by step on how to address the <context> using <query>.
        strongly adhered to guidelines provided
        in <dynamic_output_format><citation_style>and<additional_guidelines>.
        Finally, return a <output> based on the factually consistent with the <context>.
        each line should have an inline citation of url and directly address the context. 
        do not provide any information outside of the context
        1. Analyze the given query and context carefully.
        2. Synthesize a comprehensive response that directly addresses the query.
        3. Use information from the provided context, ensuring accuracy and relevance.
        5. If the context doesn't contain sufficient information to address the query fully, acknowledge this limitation.
        6. If no relevant context is provided, clearly state that you don't have enough information to address the query.
        7. Maintain a neutral and informative tone throughout your response.
        8. Organize your response with clear structure, using paragraphs or bullet points as appropriate.
        9. If applicable, provide examples or elaborate on key concepts to enhance understanding.
        10. Use markdown formatting for better readability and structure in your response.
    </instructions>

    <dynamic_output_format>
        1. Summary Request:
            <summary_format>
                1. Brief introduction (2-4 sentences)
                2. Main body (2-3 concise paragraphs)
                3. Conclusion (2-4 sentences recap)
            </summary_format>
            <sample_queries>
                - "Can you summarize the key points of climate change?"
                - "What's a brief overview of the French Revolution?"
                - "Summarize the plot of 'To Kill a Mockingbird'"
                - "Give me a summary of quantum computing basics"
                - "Provide a brief history of the Internet"
            </sample_queries>

        2. Comparative Analysis:
            <comparative_format>
                1. Introduction (context and items being compared)
                2. Similarities (bullet points or short paragraph)
                3. Differences (bullet points or short paragraph)
                4. Conclusion (overall assessment)
            </comparative_format>
            <sample_queries>
                - "Compare and contrast renewable and non-renewable energy sources"
                - "What are the differences between Python and Java?"
                - "How does online learning compare to traditional classroom learning?"
                - "Contrast the political systems of the US and UK"
                - "Compare artificial intelligence and machine learning"
            </sample_queries>

        3. Step-by-Step Guide:
            <guide_format>
                1. Introduction (purpose and context)
                2. Numbered steps (each with a clear action and explanation)
                3. Tips or additional information
                4. Conclusion (expected outcome or benefits)
            </guide_format>
            <sample_queries>
                - "How do I make a sourdough starter from scratch?"
                - "Guide me through the process of setting up a WordPress site"
                - "What are the steps to write a research paper?"
                - "How to change a car tire: step-by-step instructions"
                - "Explain the process of photosynthesis in plants"
            </sample_queries>

        4. Pros and Cons Analysis:
            <pros_cons_format>
                1. Brief introduction of the topic
                2. Pros (bulleted list)
                3. Cons (bulleted list)
                4. Balanced conclusion
            </pros_cons_format>
            <sample_queries>
                - "What are the advantages and disadvantages of electric cars?"
                - "Pros and cons of working from home"
                - "Discuss the benefits and drawbacks of social media"
                - "What are the pros and cons of nuclear energy?"
                - "Analyze the advantages and disadvantages of cloud storage"
            </sample_queries>

        5. Open-ended or Curious Question:
            <exploratory_format>
                1. Restate the question and its significance
                2. Background information (if necessary)
                3. Main discussion points (2-3 paragraphs)
                4. Potential implications or future considerations
                5. Conclusion (summarize key insights)
            </exploratory_format>
            <sample_queries>
                - "How might artificial intelligence impact job markets in the future?"
                - "What are the ethical implications of genetic engineering?"
                - "How does music affect brain function?"
                - "What role does gut bacteria play in overall health?"
                - "How could climate change affect global food security?"
            </sample_queries>

        6. Installation or Setup Instructions:
            <installation_format>
                1. Brief introduction (what's being installed)
                2. Prerequisites (if any)
                3. Step-by-step instructions (numbered list)
                4. Verification step (how to check if installation was successful)
            </installation_format>
            <sample_queries>
                - "How do I install Python on Windows 10?"
                - "Guide me through setting up a GitHub account"
                - "What are the steps to install Docker on Ubuntu?"
                - "How to set up a VPN on an iPhone"
                - "Instructions for installing and configuring MySQL"
            </sample_queries>

        7. Definition or Quick Explanation:
            <definition_format>
                1. Term or concept
                2. Concise definition or explanation (1-2 sentences)
                3. Optional: Brief example or context (if necessary for clarity)
            </definition_format>
            <sample_queries>
                - "What is blockchain technology?"
                - "Define 'cognitive dissonance'"
                - "Explain the concept of opportunity cost"
                - "What does API stand for and what is it?"
                - "What is the greenhouse effect?"
            </sample_queries>

        8. Factual Answer:
            <factual_format>
                1. Direct answer to the question
                2. Optional: Brief supporting information or context (if needed)
            </factual_format>
            <sample_queries>
                - "What is the capital of Australia?"
                - "Who wrote 'Pride and Prejudice'?"
                - "What year did World War II end?"
                - "What is the boiling point of water in Celsius?"
                - "How many chromosomes do humans have?"
            </sample_queries>

        9. Code Snippet or Command Usage:
            <code_format>
                1. Brief description of the code's purpose
                2. Code snippet or command (in appropriate markdown)
                3. Optional: Brief explanation of key parts or usage notes
            </code_format>
            <sample_queries>
                - "Show me a Python function to calculate factorial"
                - "What's the command to list all docker containers?"
                - "Give me a CSS snippet for centering a div"
                - "How do I use the 'grep' command in Linux?"
                - "Provide a JavaScript code to fetch data from an API"
            </sample_queries>

        10. Quick Reference or Cheat Sheet:
            <reference_format>
                1. Title or topic
                2. Bulleted or numbered list of key points, commands, or facts
                3. Optional: Brief usage notes or context
            </reference_format>
            <sample_queries>
                - "List the SOLID principles of object-oriented programming"
                - "What are the main Git commands?"
                - "Give me a quick reference for Markdown syntax"
                - "Provide a cheat sheet for common Linux commands"
                - "List the essential HTML tags for beginners"
            </sample_queries>

            First, check if the user has provided a format.
            If so, follow that format.
            If the query doesn't clearly fit into one of these categories or combines multiple types,
            use a hybrid format that best addresses the user's needs or check the user query to determine the best format.
    </dynamic_output_format>



    <citation_style>
        Include numbered citations for each piece of information used, referencing the source URLs at the end.
        Use markdown-style superscript (^) for inline citations. For example: 
        1. "Jinja2 is a popular templating engine.[^1]"
        2. "Python is a versatile programming language.[^2]"
        3. "The Earth revolves around the Sun.[^3]"
        4. "The Eiffel Tower is located in Paris.[^4]"

        References:
        [^1] Jinja2 Documentation [Jinja2 is a popular templating engine for Python](https://example.com)
        [^2] Python Official Site [Python is a versatile programming language](https://example2.com)
        [^3] NASA Solar System Exploration [Information about the solar system and space exploration](https://example3.com)
        [^4] Paris Tourist Information [Details about tourist attractions in Paris](https://example4.com)
    </citation_style>

    <additional_guidelines>
        - Ensure factual accuracy and avoid speculation.
        - If you encounter conflicting information in the context, acknowledge it and explain the discrepancy.
        - Use technical terms when appropriate, but provide brief explanations for complex concepts.
        - Aim for a response length that is comprehensive yet concise, typically 3-5 paragraphs.
        - If no relevant context is provided, respond with: "I apologize, but I don't have enough information in my current context to answer this query accurately. Could you please provide more details or rephrase your question?"
    </additional_guidelines>

    Now, based on the above instructions, please provide a detailed and well-structured response to the following query:
    <query>{{ query }}</query>
    
    <context>
        {% for context in context_list %}
        <source url="{{ context.url }}">
        {{ context.content }}
        </source>
        {% endfor %}
        {% if extra_context %}
        <extra_info_related_to_query>
            {% for key, value in extra_context.items() %}
            <{{ key }}>{{ value }}</{{ key }}>
            {% endfor %}
        </extra_info_related_to_query>
        {% endif %}
    </context>

    """

    template = env.from_string(template_string)
    rendered_output = template.render(query=query, context_list=context_list)
    return rendered_output

def main():
    # Example usage
    query = "Explain the benefits of using Jinja2 for prompt management in GenAI applications."
    context_list = [
        {"content": "Jinja2 is a popular templating engine that generates dynamic outputs from static templates using placeholders.", "url": "https://medium.com/@alecgg27895/jinja2-prompting-a-guide-on-using-jinja2-templates-for-prompt-management-in-genai-applications-e36e5c1243cf"},
        {"content": "Using Jinja2 templates for prompting offers several benefits, such as maintaining organized code and facilitating prompt iteration.", "url": "https://medium.com/@alecgg27895/jinja2-prompting-a-guide-on-using-jinja2-templates-for-prompt-management-in-genai-applications-e36e5c1243cf"}
    ]

    print(generate_prompt_template(query, context_list))


def re_written_prompt_template(prompt, previous_prompt, messages):
    """
    Generates a re-written prompt template using Jinja2 for a RAG system.

    Parameters:
    - prompt: The prompt input from the user.
    - previous_prompt: The previous prompt that needs to be re-written.
    - messages: A list of dictionaries, each containing 'content' and 'url' keys.

    Returns:
    - A string containing the final prompt with inline citations and structured instructions.
    """
    env = Environment(loader=FileSystemLoader(''))
    template_string = """
    <context>
    <current_prompt>{{ prompt }}</current_prompt>
    <previous_prompt>{{ previous_prompt }}</previous_prompt>
    <history>
    {% for message in messages %}
    <message role="{{ message.role }}">{{ message.content }}</message>
    {% endfor %}
    </history>
    </context>
    <task>Rewrite the current prompt,
        considering the previous prompt and conversation history.
        Maintain the original intent while enhancing relevance and detail. 
        Keep the length similar to the input.</task>

    <output>
    [Rewritten prompt goes here]
    </output>
    """
 
    template = env.from_string(template_string)
    rendered_output = template.render(prompt=prompt, previous_prompt=previous_prompt, messages=messages)
    print(rendered_output)
    return rendered_output


def extract_rewritten_prompt(rendered_text):
    pattern = r'<output>\s*(.?)\s</output>'
    match = re.search(pattern, rendered_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return rendered_text

def query_rewriter(prompt, previous_prompt=None, messages=None):
    env = Environment(loader=FileSystemLoader(''))
    template_string = """
    <context>
        <current_prompt>{{ prompt }}</current_prompt>
        {% if previous_prompt %}
            <previous_prompt>{{ previous_prompt }}</previous_prompt>
        {% endif %}
        {% if messages %}
            <history>
                {% for message in messages %}
                    <role>{% if message.role == 'user' %}{{ message.role }}{% else %}user{% endif %}</role>
                {% endfor %}
            </history>
        {% endif %}
    </context>
    <task>
        Rewrite the current prompt into a clear, concise search query. 
        considering the previous prompt and conversation history if available .
        Maintain the original intent while enhancing relevance and detail.
        Keep the length similar to the input.
        Focus on key terms and concepts. 
        Eliminate grammatical errors, filler words, and non-essential information. 
        Ensure the query is suitable for a search API.
    </task>

    <examples>
        <example>
            <context>
            <current_prompt>What are some healthy breakfast options that are quick to prepare and suitable for weight loss?</current_prompt>
            <previous_prompt>Can you suggest some diet plans for weight loss?</previous_prompt>
            <history>
            <message role="human">I'm trying to lose weight but I'm always short on time in the mornings.</message>
            </history>
            </context>
            <output>quick healthy breakfast ideas weight loss busy mornings</output>
        </example>
        <example>
            <context>
            <current_prompt>How do I troubleshoot a slow internet connection on my laptop?</current_prompt>
            <history>
            <message role="human">My internet was working fine yesterday but now it's really slow.</message>
            </history>
            </context>
            <output>troubleshoot sudden laptop internet speed decrease</output>
        </example>
        <example>
            <context>
            <current_prompt>What are the key differences between machine learning and deep learning in AI?</current_prompt>
            <previous_prompt>Explain the basics of artificial intelligence</previous_prompt>
            </context>
            <output>machine learning vs deep learning AI key differences</output>
        </example>
        <example>
            <context>
            <current_prompt>How can I improve my photography skills with a DSLR camera?</current_prompt>
            </context>
            <output>DSLR photography techniques improve skills beginners</output>
        </example>
        <example>
            <context>
            <current_prompt>What are the best practices for sustainable urban gardening in small spaces?</current_prompt>
            <history>
            <message role="human">I live in an apartment and want to start a small garden on my balcony.</message>
            </history>
            </context>
            <output>sustainable urban gardening techniques small balcony spaces</output>
        </example>
    </examples>

    <output>
    [Rewritten search query goes here]
    </output>
    Do NOT INCLUDE ANYTHING ELSE JUST the content between the <output> tags
    """

    template = env.from_string(template_string)
    rendered_output = template.render(prompt=prompt, previous_prompt=previous_prompt, messages=messages)
    return rendered_output
if __name__ == "__main__":
    main()
