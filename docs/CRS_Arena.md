# CRS Arena

The CRS Arena is a [Streamlit](https://streamlit.io) application where conversational recommender systems (CRSs) compete against each other. CRS Arena is deployed on Hugging Face Spaces and can be accessed with this [link](https://huggingface.co/spaces/iai-group/CRSArena).

The goal of this application is for a user to have a conversation with two CRSs and vote on which one they prefer. All the conversations are recorded and saved for future analysis. When the user arrives on the app, they are assigned a unique ID. Then, two CRSs are chosen for the battle depending on the number of conversations already recorded (i.e., the CRSs with the least number of conversations are chosen). The user interacts with the CRSs one after the other. The user can then vote on which CRS she/he prefer, upon voting a pop-up will appear giving the user the option to provide a more detailed feedback. Once the vote is submitted, all data is logged and the app restarts for a new battle.

The application comprises five sections:

1. **Title & Introduction**: A brief introduction to the application.
2. **Rules**: A short list of rules to follow in the arena.
3. **Side-by-Side Battle**: This is the main component. It displays two chat interfaces side by side, i.e., one for each CRS. A chat interface comprises a chat window (messages and input box) and two buttons to express the feeling at the end of the conversation, i.e., frustrated or satisfied.
4. **Vote**: The user decides on a winner or declares a tie. The vote is followed by a pop-up to optionally provide more detailed feedback.
5. **Terms of Service**: A brief description of the terms of service.

As of now, the CRS Arena only supports conversational recommender systems making movie recommendations.

## Conversational Recommender Systems

Currently, the CRS Arena supports the following CRSs:

| CRS | Training/Supporting Dataset |
| --- | --------------------------- |
| KBRD | RedDial |
| KBRD | OpenDialKG (subset incl. movies and books) |
| UniCRS | RedDial |
| UniCRS | OpenDialKG (subset incl. movies and books) |
| BARCOR | ReDial |
| BARCOR | OpenDialKG (subset incl. movies and books) |
| ChatGPT | ReDial |
| ChatGPT | OpenDialKG (subset incl. movies and books) |
| CRB-CRS | ReDial |

The CRS configurations are stored in `data/arena/crs_config/`. Each configuration is a YAML file containing the CRS arguments. For details on the CRS arguments, please refer to the respective CRS under `src/model/`.

## Data collection

Two types of data are collected in the CRS Arena:

1. Conversations between the user and the CRSs.
2. User vote and feedback on the CRSs.

### Conversations

The conversations and the user sentiment (i.e., frustated or staisfied) after each conversation are stored in a JSON file as follows:

```json
[
    {"role": "user", "message": "Hello"},
    {"role": "system", "message": "Hi! How can I help you today?"},
    {"role": "user", "message": "I am looking for a movie"},
    {"role": "metadata", "sentiment": "satisfied"}
]
```

Note that the last entry is not part of the conversation but is used to store the user sentiment.

The JSON files are both saved locally in `data/arena/conversation_logs/` and uploaded to a private Hugging Face dataset. For more information on Hugging Face datasets, see the [documentation](https://huggingface.co/docs/datasets/).

### User vote and feedback

The user vote and feedback are stored in a private Google Sheet. The votes are stored in the following format:

| timestamp | user_id | crs_1 | crs_2 | vote |
|-----------|---------|-------|-------|------|
| 2021-10-01 12:00:00 | 2d310c16-580f-42c1-90a5-cb8661861656 | CRS1_Name | CRS2_Name | CRS2_Name |

The feedback is stored in the following format in a separate sheet:

|timestamp | feedback |
|----------|----------|
| 2021-10-01 12:00:00 | Optional feedback |

## Run the application locally

The application can be run locally using the following command:

```bash
python -m streamlit run crs_arena/arena.py
```

Note that to run the application locally, you need to create the required secrets in `.streamlit/secrets.toml`.

### Create secrets

The secrets are stored in the file `.streamlit/secrets.toml`. Below you have a template of the file with placeholders for the secrets. You need to replace the placeholders with your secrets.

```toml
[openai]
api_key = "{YOUR_OPENAI_API_KEY}"

[files]
models_folder_url = "https://gustav1.ux.uis.no/crs_arena/models.tar.gz"
item_embeddings_url = "https://gustav1.ux.uis.no/crs_arena/embed_items.tar.bz2"

[hf]
dataset_repo = "{YOUR_HF_DATASET_REPO}"
hf_user = "{YOUR_HF_USERNAME}"
hf_token = "{YOUR_HF_TOKEN}"

[connections.gsheets]
{GOOGLE_SHEET_CONNECTION_PARAMS}
```

Refer to the [documentation of Streamlit GSheetsConnection](https://github.com/streamlit/gsheets-connection) for the `{GOOGLE_SHEET_CONNECTION_PARAMS}`. :warning: You need a private Google Sheet to store the user votes and feedback.
