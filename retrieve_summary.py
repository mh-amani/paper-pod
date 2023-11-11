from openai import OpenAI


def retrieve_summary(paper_content, prompt_header=None, num_pages=1, model="gpt-4-1106-preview"):
    if prompt_header is None:
        prompt_header = f"Read this paper and generate a critical {num_pages}-page summary of the keypoints. \
                make sure to include the following information in your summary if available: \
                - the inputs and outputs of the machine learning model. \
                - the datasets used to train and evaluate the model. \
                - the model architectures, novelties and the training methods. \
                - the results and the evaluation metrics. \
                - the limitations of the model. \
                remember, it is a critical summary, you are supposed to describe the paper in the position of one challenging the paper. \
                you can also include your own opinions and ideas."
    
    client = OpenAI()
    response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": prompt_header},
        {"role": "user", "content": paper_content},
    ]
    )

    return response.choices[0].message.content