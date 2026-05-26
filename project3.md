# Project 3: Build a Retrieval Augmented Generation (RAG) System

**Project Instruction**  
This is a group project. Students should form teams of 2–3 members. All group members are expected to contribute meaningfully to the project and share responsibility for the final submission.

**Deadline:** 2026-6-14 23:59 (Late submissions will be penalized by 1 point per day)  
**Submission Format:** Final report must be in PDF. Code, video and test results must be zipped as: studentID-name-project3.zip  
**Submission Link:**
- upload zip file to https://epan.shanghaitech.edu.cn/l/zFFGZ9
- upload report to https://ecourse.shanghaitech.edu.cn/  

**GPU Resources:** https://aistation2.shanghaitech.edu.cn:32206/ (40GB GPU memory each group)

## 1. Objective
Your team will design, implement, and deploy a Retrieval-Augmented Generation (RAG) system that can accurately answer questions about **ShanghaiTech University** and the **School of Information Science and Technology (SIST)**. Example queries include:
- 上海科技大学一共有几个学院？
- 《深度学习》这门课的任课老师是谁？
- 计算机科学与技术专业需要修满多少学分才能毕业？

The system can be deployed on the AI platform (https://aistation2.shanghaitech.edu.cn:32206/) or your own server with an interactive web interface (e.g., using Gradio).
## 2. Steps  

### 2.1 Data Collection & Preprocessing
- We provide a basic dataset that contains only part of the information about ShanghaiTech and SIST:
    - https://epan.shanghaitech.edu.cn/v/link/view/1ffaff1559e444bfa50862cfc71b2608
- Teams are expected to **collect additional relevant data** from official sources (e.g., school websites, course catalogs, faculty pages, news articles, etc.) to enhance coverage and accuracy.
- Implement a data preprocessing pipeline to format, clean, chunk, and prepare the text for embedding and retrieval.
- Your raw data might include a mix of HTML pages, PDFs, and plain text documents. You will need to clean this data and convert it into a file format that suites your development. Here are some tools that you could use:
  - **PDF to Text**: Use tools like `pdftotext`, `pdfplumber` or `PyPDF2` to extract text from PDFs.
  - **HTML to Text**: Use tools like `beautifulsoup4` to parse HTML and extract text.

### 2.2 RAG System Implementation
- Build a complete RAG pipeline consisting of:
  - A retriever (e.g., using dense embeddings like sentence-transformers or sparse methods like BM25).
  - A generator (an open-source LLM like Qwen). 
- Implement the core logic to retrieve relevant context from your knowledge base and generate answers based on it.
- (optional) Enable the system to intelligently decide whether to retrieve relevant context or generate an answer directly based on the input.
- The entire RAG system must rely solely on self-deployed or locally running models, and must **not** use any commercial LLM APIs.

### 2.3 Interactive Web Interface & Deployment
- Create an intuitive user interface using a framework like Gradio or Streamlit.
- The interface should allow users to input a question and display the generated answer along with the retrieved source context.
- Deploy your application on the AI platform or your own server. 

### 2.4 Evaluation, Analysis, and Optimization
- Design a set of test questions (at least 50, preferably more) covering different aspects of ShanghaiTech/SIST, and different types of questions, such as:
  - Factual: 上海科技大学信息学院成立于哪一年？
  - Multi-hop: 《深度学习》这门课任课老师的学术背景？
  - Time-sensitive: 上海科技大学最新的讲座信息
  - Comparative: 信息学院专业型硕士与学术性硕士的培养方案有什么不同？
  - Conditional: 我想做机器人方向，有哪些导师可以推荐？
  - And more...
- Evaluate your system's performance using relevant metrics such as accuracy, efficiency, or others you consider important for your system.
- Analyze failure cases and discuss their causes.
- Implement at least one optimization technique to improve either accuracy (e.g., query expansion, re-ranking) or efficiency (e.g., retrieval optimization, context compression). Compare the performance before and after the optimization in your report.
- **Note**: During grading, the quality of your test set will be taken into account.

## 3. Report Requirements
Your PDF report must include the following sections:
- **Introduction**: Briefly state the project's goal and your team's approach.
- **Data Collection & Preprocessing**: Detail your data sources (beyond the provided dataset), your preprocessing steps, and the final structure of your knowledge base.
- **System Architecture**: 
    - Describe the entire RAG pipeline, from user query input to final answer output.
    - Provide a clear diagram illustrating the system architecture.
- **Implementation Details**: 
    - **Components**: Describe each component in your system in detail, including embedding model, vector database, LLM and any other key components or libraries you employed.
    - **Technical Details**: Describe your technical details, such as query handling, retrieval strategy and the prompt template used for the LLM.
    - **Optimization**: Clearly explain the specific optimization technique you implemented.
    - **Challenges**: Discuss any significant challenges you encountered during implementation and how you resolved them.
- **Results**: 
  - Show examples of successful and failed queries with analysis.
  - Present a comparison of relevant metrics (e.g., accuracy, latency) before and after your optimization.
  - Include 3-5 representative case studies that showcase your system's capabilities and the impact of your optimizations.
- **Discussion & Conclusion**: Summarize your key findings, the effectiveness of your approach, limitations, and potential future work.
- **References**: Cite any external resources, models, or papers you used.

## 4. Deliverables
- **Source code**
    - All code must be well-structured, modular, and commented.
    - Include a README.md file with clear instructions for deployment.
    - Do not include the raw dataset files or large model checkpoints in your submission.
- **A technical report** (in English, PDF format)
- **Test results**: Provide your test results on your test set in **an Excel file**, with columns including:
    - **query**: the test question
    - **gt_answer**: ground truth answer
    - **sys_resp_before_opt**: system response before optimization
    - **sys_resp_after_opt**: system response after optimization
    - **is_correct_before_opt**: 1 if the answer is correct, 0 otherwise
    - **is_correct_after_opt**: 1 if the answer is correct, 0 otherwise
- **presentation**: 
    - Due to limited class time in Week 16, only some groups will be selected to present live; the rest must submit a pre-recorded video. Each presentation should be about 10 minutes long.
    - All presentations—live or recorded—must include:
        - A detailed overview of your key work, including system design, main optimizations, and evaluation results.
        - A demo of your user interface, showing your system's capabilities.
- The test results file, presentation video, and code must be zipped together as studentID-name-project3.zip.

## 5. Resources
Faiss: https://github.com/facebookresearch/faiss/wiki
Sentence Transformers Documentation: https://www.sbert.net/
Gradio: https://www.gradio.app/
[ACL 2023 Tutorial: Retrieval-based Language Models and Applications](https://acl2023-retrieval-lm.github.io/)
Gao et al., 2023. [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997).
Lewis et al., 2021. [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401).
