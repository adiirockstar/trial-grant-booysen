# Ubundi Trial Project – Grant's Personal Codex Agent

## 🚀 Overview
This project is my lightweight "personal codex agent" for Ubundi’s trial project.  
It ingests my CV and supporting documents, then lets you ask natural language questions about me.  

## Setup
1. Clone this repo
2. Add your documents into `/data`
3. Install requirements:
   ```bash
   pip install -r requirements.txt

## System Setup

A dynamic RAG (Retrieval-Augmented Generation) system that creates an AI assistant based on Grant Booysen's personal and professional documents. The system allows users to upload documents and interact with Grant's knowledge base through different response modes.

## 🌟 Features

### Core Functionality
- **Multi-Modal Responses**: 6 different response styles (Interview, Story, FastFacts, HumbleBrag, Reflect, Project)
- **Dynamic Document Upload**: Add new training data through web interface
- **RAG-Powered Q&A**: Intelligent retrieval of relevant context for accurate responses
- **Real-time Processing**: Instant document processing and knowledge base updates

### Document Support
- **File Formats**: TXT, PDF, DOCX, Markdown
- **Multiple Upload**: Batch upload multiple files
- **In-Memory Processing**: Cloud-compatible, no permanent storage required
- **Smart Chunking**: Automatic text segmentation for optimal retrieval

### User Experience
- **Interactive Sidebar**: Easy mode switching and document management
- **Quick Questions**: Pre-defined buttons for common queries
- **Session Management**: Clear uploads, refresh knowledge base
- **Download Backup**: Export session data for safekeeping

## System Architecture

```
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   Streamlit UI      │    │   RAG Pipeline       │    │   Groq API          │
│                     │    │                      │    │                     │
│ • Mode Selection    │◄──►│ • Document Loading   │◄──►│ • Llama3-8B Model   │
│ • File Upload       │    │ • Text Chunking      │    │ • Response Generation│
│ • Q&A Interface     │    │ • Vector Embeddings  │    │ • Context Integration│
│ • Session Management│    │ • Similarity Search  │    │                     │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
                                      │
                           ┌──────────────────────┐
                           │   FAISS Vector DB    │
                           │                      │
                           │ • In-Memory Storage  │
                           │ • Semantic Search    │
                           │ • Real-time Updates  │
                           └──────────────────────┘
```

## Sample questions and expected answers
- Question: Tell me about yourself? How would your personal values impact your work?
- Answer: Speaks about my desire for responsible and ethical development. Speaks about the projects that Ubundi has to offer that I personally align with. 

- Question: What are your core technical skills and strengths?
- Answer: Mention coding languages, computer science skills, experience in reinforcement learning and overall how I would be a good candidate for Ubundi.

- Question: What projects are you most proud of?
- Answer: Expected answer should include the Recommender System I am building, any reinforcement learning project that I have completed. The bot sometimes answers that I am a proud user of DALLE which dodges the question a little.

## In-Memory data storage and online learning
Since I have a real time update feature I encourage you to see how it works! I have uploaded a test file for you to experiment with under the directory `/build_artefacts/temp.txt`. In there describes my love for animals. Upload the file to the live session storage and see the results.

- Question: What is your dream?
- Before: My dream is to create a tool that truly makes a tangible difference in people's lives. I've had the privilege of working on some amazing projects, but Horizon, my proposed AI guide, is something that I'm truly passionate about. I believe that with Horizon, I can help individuals who have been through intense or isolated experiences, such as those in correctional facilities, the military, or law enforcement, rediscover their place in society and find a sense of purpose again.

As I see it, Horizon would be more than just a tool - it would be a companion that learns about each individual's unique experiences, values, and aspirations, and provides them with tailored suggestions and support to help them heal and reconnect with the world. I envision Horizon being a reflection of the user's identity, amplifying their personal story and intentions. It's a product that would empower individuals to take control of their lives and find their path forward.

For me, this dream is not just about creating a technology, but about being part of a movement that promotes healing, connection, and growth. I'm excited about the prospect of being part of a team that shares this vision and is committed to making a positive impact on people's lives.

- After: You know, I've always had a deep passion for wildlife conservation, and if I'm being completely honest, it's a dream of mine to become a park ranger one day. I grew up camping in reserves like Sondela, and it was always an incredible experience for me. I think what draws me to it is the sense of peace and fulfillment I get from being surrounded by nature. But, as much as I love animals, I know my true calling is solving problems and making a real difference in people's lives. If I had to choose a job that would bring me peace, it would definitely be working in wildlife conservation.

## What would I improve with more time
- I would make modes for different companies so that the bot does not answer with Ubundi related answers for all users.
- I would connect to different API's when the prompt returns a personal link so that the user can interact with my interests a bit more. For example, when asking about my worship music, the bot should recognise that I have a worship playlist with a link, then post a request to the Spotify endpoint to display it in the streamlit app so that you can view the songs in streamlit.
- I would make a Usage Analytics to keep logs of how frequently certain questions are getting asked that I would keep for myself. This would allow me to keep track of which answers need improvement so that I can upload the necessary data and input to keep the bot relevant.
- I would make a calendar intergration so that if you are interested in an interview, we can schedule an appointment.

## What to keep note of
The side bar on the left has a lot of functionality. I encourage the tester to try out the VERY bottom section - in my opinion, it is quite cool.

## Show thinking artefacts
I have kept a few of the prompt questions that I have asked ai bots to help build this project under the directory `/build_artefacts/`. Since the bots return with very lengthy answers I gave the most important prompts in the summary, and some code snippets in the conversations.md.