import React from 'react';
import { render } from '@testing-library/react';
import MessageList from '../src/components/MessageList';
import type { Message } from '../src/components/MessageItem';

describe('MessageList', () => {
  it('shows newest message first', () => {
    const messages: Message[] = Array.from({ length: 30 }, (_, i) => ({
      id: String(i),
      role: 'bot',
      text: `msg${i}`,
    }));
    const { container } = render(<MessageList messages={messages} />);
    const first = container.firstElementChild?.firstElementChild as HTMLElement;
    expect(first.textContent).toBe('msg29');
  });
});
