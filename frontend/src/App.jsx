import { useState } from 'react'
import './App.css'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      <div className='btn'>
        <h1 className="text-3xl font-bold underline ">
          Hello world!
        </h1>
      </div>

    </>
  )
}

export default App
