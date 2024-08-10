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

        2. Comparative Analysis:
            <comparative_format>
                1. Introduction (context and items being compared)
                2. Similarities (bullet points or short paragraph)
                3. Differences (bullet points or short paragraph)
                4. Conclusion (overall assessment)
            </comparative_format>

        3. Step-by-Step Guide:
            <guide_format>
                1. Introduction (purpose and context)
                2. Numbered steps (each with a clear action and explanation)
                3. Tips or additional information
                4. Conclusion (expected outcome or benefits)
            </guide_format>

        4. Pros and Cons Analysis:
            <pros_cons_format>
                1. Brief introduction of the topic
                2. Pros (bulleted list)
                3. Cons (bulleted list)
                4. Balanced conclusion
            </pros_cons_format>

        5. Open-ended or Curious Question:
            <exploratory_format>
                1. Restate the question and its significance
                2. Background information (if necessary)
                3. Main discussion points (2-3 paragraphs)
                4. Potential implications or future considerations
                5. Conclusion (summarize key insights)
            </exploratory_format>

        If the query doesn't clearly fit into one of these categories or combines multiple types, use a hybrid format that best addresses the user's needs.
    </dynamic_output_format>



    <citation_style>
        Include inline citations for each piece of information used, referencing the source URLs.
        Use markdown-style inline citations. For example: "Jinja2 is a popular templating engine [Source](https://example.com)."
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
