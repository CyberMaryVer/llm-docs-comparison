# **Summary**:

## Comparing Approaches: Base vs. RAG

**Base Approach**
- Analyze each chunk
- Extract meaningful info
- Combine altogether into summary with key points
- Compare two documents by their summaries

**RAG Approach**
- Split both documents to chunks with overlapping
- Vectorize Doc 2 to faiss index
- Iterate chunk by chunk through Doc 1, for each chunk find the nearest one in Doc 2
- Compare 2 chunks with LLM, extract only key differences
- Summarize all differences

When comparing two documents, two approaches can be employed: the Base approach and the Retrieval-Augmented Generation (RAG) approach. Here are the key takeaways from exploring both methods:

### Base Approach
**Pros:**

- Simplicity and Robustness: The Base approach is more straightforward and robust, making it a suitable option for language models (LLMs) with a very large context window.
- Consistency: This approach processes documents in a single, linear fashion, ensuring consistent results.

**Cons:**

- Time-Consuming: It takes a significant amount of time to process (24 minutes in this case).
- High Token Usage and Cost: The method requires more tokens, leading to higher processing costs.
- Lacks Explainability: The approach lacks intermediate steps, which can make it challenging to validate results or understand the reasoning behind the outcomes.

**How can be improved**

- Key Points List: Prepare a list of main key points to identify for each type of document. Summarize the document chunk by chunk, extracting information according to this list. This will significantly reduce cost and probably increase accuracy for these key issues.

  FYI: This method may face problems if there is a big variety of possible key issues. 

### RAG Approach
**Pros:**

- Intermediate Results: The RAG approach consists of several steps, providing intermediate results that can be used for validation and better explainability.
- Faster and Cost-Effective: This method is faster and cheaper because it doesn't require a summary of the entire document. It is also more flexible, allowing the use of cheaper LLMs for processing and information extraction, with more powerful models like GPT-4 reserved for final summarization.
- Flexibility: The RAG approach offers the ability to mix and match different models for different stages of the process, optimizing both performance and cost.

**Cons:**

- Complexity: The RAG approach is more complicated due to its multi-step process, which might require more sophisticated implementation and management.
- Basic version of RAG presented here less accurate, comparing to results with the base approach.

**How can be improved**
- Use a better LLM: The first extraction step can be switched to GPT-4, which will be more costly but also more accurate.
- Implement validation steps for responses to ensure higher accuracy and reliability of the results.
- Change the retrieval method to increase accuracy, even if it slightly increases costs.
- Implement a 'differences with sources' method to return page numbers or paragraph numbers.
- Implement more strict conditions for classifying differences as meaningful.

In summary, the Base approach is simple and robust but time-consuming and costly, with limited explainability. The RAG approach, while more complex, offers better explainability, faster processing, and cost-effectiveness by leveraging intermediate results and flexible model usage.

However, the final approach should be chosen according to test results with experts and aligned with business goals.

