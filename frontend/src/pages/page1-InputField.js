import React, {useState} from "react";
import {Box, Button, TextField} from "@mui/material";

const InputField1 = ({onSend1}) => {
    const [text1, setText1] = useState('');

    const handleChange1 = (event) => {
        setText1(event.target.value);
    };

    const handleSendClick1 = () => {
        if (text1.trim() !== '') {
            onSend1(text1);
            setText1('');
        }
    };

    return (
        <Box className={'input-container1'}> 
            <TextField
                className={'input-field1'}
                placeholder='メッセージを入力してください'
                value={text1}
                onChange={handleChange1}
            />
            <Button
                className={'send-button1'}
                variant='contained'
                onClick={handleSendClick1}
            >
                送信
            </Button>
        </Box>
    );
};

export default InputField1;