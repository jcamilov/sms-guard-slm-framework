prompts = {
    "prompt_01": {
        "prompt":"""You are a cybersecurity expert. Analyze the following SMS and classify it as 'smishing' or 'benign':
    #Input
    {sms_text}

    #Output
    Do not reprint the SMS. Only respond in the following format:

    ##Classification: 'smishing' or 'benign'
    ##Explanation: For example: 'The SMS contains a suspicious link and requests for personal information.'
    """,
    "description": "Without format example"
    },
    
    "prompt_02": {
        "prompt":"""Classify the following as 'smishing' or 'benign':
    #Input
    {sms_text}

    #Output
    Do not reprint the SMS. Only respond in the following format:

    ##Classification: 'smishing' or 'benign'
    ##Explanation: For example: 'The SMS contains a suspicious link and requests for personal information.'

    #Example
    Input: "Your account has been compromised. Please click the link to verify your account. https://bit.ly/3QaSxy4"
    Output: Classification: 'smishing'
    Explanation: The SMS contains a suspicious link and requests for personal information.
    """,
    "description": "With format example"
    }
} 