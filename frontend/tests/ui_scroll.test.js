try { require.resolve('react'); } catch { test.skip('React/Jest deps missing', () => {}); }
require('@babel/register')({ presets: ['@babel/preset-react'] });
const React = require('react');
const { render } = require('@testing-library/react');
const MessageList = require('../src/components/MessageList').default;

test('firstChild shows last added message', () => {
  const msgs = [];
  const { rerender, container } = render(React.createElement(MessageList, { messages: msgs }));
  msgs.unshift({ id: '1', role: 'bot', text: 'hello' });
  rerender(React.createElement(MessageList, { messages: msgs }));
  expect(container.firstChild.firstChild.textContent).toBe('hello');
});
