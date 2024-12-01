import React, {useState} from "react";
import {Box} from "@mui/material";
import InputField1 from "./page1-InputField";
import MessageList1 from "./page1-MessageList";


const Page1 = () => {  
    const [messages1, setMessages1] = useState([]);  
    
    const handleSend = async (text1) => {  
        setMessages1((prevMessages) => [  
        ...prevMessages,
        { content1: text1, role: 'user' },
        ]);      
    
    
    try{
        const response = await fetch('http://127.0.0.1:5000/message', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({message: text1}),
        });
        const data = await response.json();

        setMessages1((prevMessages) => [  
        ...prevMessages,
        { content1: data.response, role: 'assistant' },
        ]);
    }
    catch (error) {
        console.error('Error:', error);

        setMessages1((prevMessages) => [
        ...prevMessages,
        { content1: 'エラーが発生しました。', role: 'assistant' },
        ]);
    }
};
    
    
    return ( 
        <Box className={'Page1'}>
            <Box className={'header1'}>
                <button className={'header1-button'} type="button" onClick={() => window.location.href = '/'}>ホームへ</button>
                <p className={'header1-p'}>オタク女子</p>
            </Box>
            <MessageList1 messages1={messages1} />
            <InputField1 onSend1={handleSend} />
        </Box>
    );
};

export default Page1;