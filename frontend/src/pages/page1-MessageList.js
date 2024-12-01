import {Box} from "@mui/material";
import React, {useEffect, useRef} from "react";
import otakujoshi from './images/22996495.png'; 

const Message1 = ({content1, role}, index) => {
    return (
        <Box className={'message-1'}>
            <Box>
                {role === 'assistant' && (<img src={otakujoshi} className={'message-img1'} alt={'assistant'} />)}
            </Box>
            <Box key={index} className={`message-page1 ${role}`}>
                {content1}
            </Box>
        </Box>
    );
};


const MessageList1 = ({messages1}) => {

    const messageListRef = useRef(null);

    useEffect(() => {
        if (messageListRef.current) {
            console.log('scrolling to bottom');
            messageListRef.current.scrollIop = messageListRef.current.scrollHeight;
        }
    }, [messages1]);

    return (
        <Box className={'message-list1'} ref={messageListRef}>
            {messages1 && messages1.map((messages1, index) => (
                <Message1 key={index} content1={messages1.content1} role={messages1.role} />
            ))}
        </Box>
    );
};

export default MessageList1;