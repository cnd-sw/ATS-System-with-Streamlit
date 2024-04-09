# ATS-System-with-Streamlit

# Resume-and-CV-ATS
Summarising the resume and then running ATS test on it and then accepting or rejecting the applicant and stating the reason for rejection.

In the project, the ATS (Applicant Tracking System) score was calculated using a combination of keyword matching and scoring techniques. The goal was to assess the relevance of a resume to a specific job or category based on the presence of relevant keywords.

Here's how the ATS score calculation process works:

1. **Keyword Identification**: Relevant keywords or key phrases are identified for each job or category. These keywords represent the skills, qualifications, experience, or other attributes desired for the role.

2. **Text Analysis**: The resume text is analyzed to identify the presence of these relevant keywords. This may involve techniques such as tokenization, text preprocessing, and keyword matching.

3. **Keyword Matching**: Each keyword found in the resume contributes to the ATS score. The more relevant keywords a resume contains, the higher its ATS score. Some variations in scoring can also consider factors such as the frequency of keyword occurrences or their importance.

4. **Scoring**: The presence of each relevant keyword in the resume contributes to the overall ATS score. The score can be calculated based on various criteria, such as a binary score (1 for presence, 0 for absence), weighted score (assigning different weights to keywords based on importance), or other scoring mechanisms.

5. **Normalization**: Optionally, the ATS score can be normalized to a scale of 0 to 100 for easier interpretation. This allows for a more standardized comparison of different resumes.

6. **Threshold**: A threshold value is set to determine whether a resume is considered suitable or "shortlisted" based on its ATS score. Resumes with ATS scores above the threshold are shortlisted for further consideration, while those below the threshold may be rejected.

7. **ATS Evaluation**: Finally, the ATS score is evaluated against the threshold to determine the shortlisting status of the resume. If the ATS score exceeds the threshold, the resume is considered suitable and may be shortlisted for further review. Otherwise, it may be rejected.

Regarding the project's architecture, here's a high-level overview:

1. **Data Collection**: Resumes are collected from various sources and stored in a dataset.

2. **Preprocessing**: The resume data undergoes preprocessing steps to clean and standardize the text data, including tasks like removing irrelevant information, standardizing formatting, and tokenizing the text.

3. **Keyword Identification**: Relevant keywords are identified for each job or category. These keywords are used as input for the ATS score calculation.

4. **ATS Score Calculation**: The resume text is analyzed, and the ATS score is calculated based on the presence and relevance of keywords using the algorithm described above.

5. **Model Integration**: The ATS score calculation algorithm may be integrated into a larger system that includes other components, such as classification models for categorizing resumes into specific job roles or categories.

6. **User Interface**: A user-friendly interface is developed to allow users to input their resume text and view the ATS score and shortlisting status.

7. **Deployment**: The system, including the ATS score calculation algorithm and user interface, is deployed to a server or cloud platform where it can be accessed by users.

👉🏻**Use of classification Algorithms**

In this project, classification algorithms were used for the following reasons.

1. Automated Screening: Manual screening of resumes is time-consuming and prone to human bias. By using classification algorithms, the screening process can be automated, saving time and reducing human bias.

2. Categorization: Resumes often belong to different categories (e.g., software engineering, data science, marketing). Classification algorithms can categorize resumes into these predefined categories based on their content.

3. Keyword Extraction: Classification algorithms can analyze the content of resumes and extract relevant keywords or features that are indicative of the category to which the resume belongs. These keywords or features can then be used to make classification decisions.

4. Scalability: As the number of resumes to be screened increases, manual screening becomes increasingly impractical. Classification algorithms can handle large volumes of resumes efficiently, making the screening process scalable.

5. Consistency: Classification algorithms provide consistent results based on predefined criteria, reducing variability compared to manual screening by different individuals.


👉🏻 **How was ATS calculated**

In this project, the Applicant Tracking System (ATS) score was calculated as a metric to evaluate the suitability of a resume for a particular job or category. The ATS score indicates the degree to which a resume matches the job requirements or relevant keywords.

Here's an overview of how the ATS score was calculated in the project:

1. Keyword Matching: The first step in calculating the ATS score is to identify relevant keywords or terms that are indicative of the desired skills, qualifications, or experience for the job or category.

2. Text Analysis: The resume text is then analyzed to identify the presence of these relevant keywords. This can be done using various techniques such as tokenization, text preprocessing, and keyword matching.

3. Scoring: Each keyword found in the resume contributes to the ATS score. The more relevant keywords a resume contains, the higher its ATS score. Some variations in scoring can also consider factors such as the frequency of keyword occurrences or their importance.

4. Normalization: Optionally, the ATS score can be normalized to a scale of 0 to 100 for easier interpretation. This allows for a more standardized comparison of different resumes.

5. Threshold: A threshold value is typically set to determine whether a resume is considered suitable or "shortlisted" based on its ATS score. Resumes with ATS scores above the threshold are shortlisted for further consideration, while those below the threshold may be rejected.

6. ATS Evaluation: Finally, the ATS score is evaluated against the threshold to determine the shortlisting status of the resume. If the ATS score exceeds the threshold, the resume is considered suitable and may be shortlisted for further review. Otherwise, it may be rejected.
