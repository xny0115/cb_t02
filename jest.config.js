module.exports = {
  testEnvironment: 'jsdom',
  roots: ['<rootDir>/frontend/tests'],
  transform: {
    '^.+\\.jsx?$': 'babel-jest'
  }
};
