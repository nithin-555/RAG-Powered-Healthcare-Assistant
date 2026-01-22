def build_prompt(query, context_items):
    """
    Constructs a prompt for the LLM using the retrieved context.
    """
    context_str = ""
    for i, item in enumerate(context_items):
        context_str += f"Source {i+1} (Focus: {item.get('Focus', 'GeneralResponse')}):\n"
        context_str += f"Q: {item['Question']}\n"
        context_str += f"A: {item['Answer']}\n\n"
        
    prompt = f"""You are a trusted healthcare assistant. Use the following medical context to answer the user's question.
If the answer is not contained in the context, say "I don't have enough information in my knowledge base to answer this accurately." and advise them to consult a doctor.
Do not hallucinate information.

User Question: {query}

Context:
{context_str}

Answer:"""
    return prompt
