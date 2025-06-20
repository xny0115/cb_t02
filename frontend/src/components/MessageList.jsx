import React from 'react';
import MessageItem from './MessageItem';

export default function MessageList({ messages }) {
    return (
        <ul className="flex flex-col-reverse overflow-y-auto h-full min-h-0">
            {messages.slice().reverse().map(m => (
                <MessageItem key={m.id} {...m} />
            ))}
        </ul>
    );
}
