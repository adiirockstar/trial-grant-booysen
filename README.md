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