from typing import Dict, List

class ChatHistory:
    """Simple chat history management"""
    def __init__(self, max_history: int = 10):
        self.history: Dict[str, List[Dict]] = {}  # user_id -> list of messages
        self.max_history = max_history
    
    def add_message(self, user_id: str, role: str, content: str):
        """Add a message to user's history"""
        if user_id not in self.history:
            self.history[user_id] = []
        
        self.history[user_id].append({"role": role, "content": content})
        
        # Trim history if too long
        if len(self.history[user_id]) > self.max_history:
            self.history[user_id] = self.history[user_id][-self.max_history:]
    
    def get_context(self, user_id: str, last_n: int = 3) -> str:
        """Get recent chat history as context string"""
        if user_id not in self.history or len(self.history[user_id]) < 2:
            return ""
        
        # Get last n exchanges (n pairs of user/assistant messages)
        messages = self.history[user_id][-(last_n * 2):] if last_n > 0 else []
        
        context_parts = []
        for msg in messages:
            speaker = "User" if msg["role"] == "user" else "Assistant"
            context_parts.append(f"{speaker}: {msg['content']}")
        
        return "\n".join(context_parts)
    
    def clear(self, user_id: str):
        """Clear user's history"""
        if user_id in self.history:
            self.history[user_id] = []