import React from 'react';
import MessageItem, { Message } from './MessageItem';

interface Props {
  messages: Message[];
}

export default function MessageList({ messages }: Props) {
  return (
    <ul id="chatMessages" className="flex flex-col-reverse overflow-y-auto h-full min-h-0">
      {[...messages].reverse().map(m => (
        <MessageItem key={m.id} {...m} />
      ))}
    </ul>
  );
}
