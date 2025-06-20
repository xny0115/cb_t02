import React from 'react';
import MessageItem from './MessageItem';

export default function MessageList({ messages }) {
  return React.createElement(
    'ul',
    { id: 'chatMessages', className: 'flex flex-col-reverse overflow-y-auto h-full min-h-0' },
    messages.map(m => React.createElement(MessageItem, { ...m, key: m.id }))
  );
}
