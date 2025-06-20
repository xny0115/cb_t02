#!/usr/bin/env node
const { execSync } = require('child_process');
function has(pkg){
  try{ require.resolve(pkg); return true; }catch{ return false; }
}
if(!(has('react') && has('jest'))){
  try{ execSync('npm install', { stdio: 'inherit' }); }catch(e){}
}
