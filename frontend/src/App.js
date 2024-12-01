import './App.css';
import React from 'react';
import {Routes, Route} from 'react-router-dom';
import Home from './pages/Home';
import Page1 from './pages/page1';
import Page2 from './pages/page2';


import './css/Home.css';
import './css/page1.css';
import './css/page2.css';


const App = () => {
  return (
    <div className='App'>
      <Routes>
        <Route path='/' element={<Home />} />
        <Route path='/Page1' element={<Page1 />} />
        <Route path='/Page2' element={<Page2 />} />
      </Routes>
    </div>
  );
};

export default App;