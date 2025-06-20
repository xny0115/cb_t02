import React, { useState } from 'react';
import TrainingStatusBar from './components/TrainingStatusBar';
import MessageList from './components/MessageList';

export default function App() {
  const [messages, setMessages] = useState([]);

  const addMessage = (msg) => setMessages(prev => [msg, ...prev]);

  return (
    <div className="h-full flex flex-col">
      <TrainingStatusBar />
      <MessageList messages={messages} />
    </div>
  );
}
