import { useNavigate } from "react-router-dom";
import otakujoshi from './images/22996495.png';


const Home = () => {

    const navigate = useNavigate();
    const Page1Form = () => {
        navigate('/Page1');
    }

    const Page2Form = () => {
        navigate('/Page2');
    }

    return (
        <div className={'home'}>
            <h1 className={'hometitle'}><span className={'hometirle-sub'}>推し</span>活チャットボット</h1>

            <div className={'chatbot'}>
                <div className={'chatbot1'}>
                    <div className={'chatbot1-name'}>
                        <img src={otakujoshi} className={'joshi-logo'}></img>
                        <p>オタク女子</p>
                    </div>
                    <p className={'chatbot1-exp'}>オタク女子です。特に推しキャラに対する愛がとても深く、<br/>
                        似たような趣味を持つ人との会話を楽しみにしています。<br/>
                        推しについて私と、とことん語り合いましょう。</p>
                    <button className={'chatbot1-but'} type='button' value='Page1' onClick={Page1Form}>クリック</button>
                </div>

                <div className={'qestion'}>
                    <p className={'qestion-text'}>何を質問すればいいか分からない人はここ⇒</p>
                    <button className={'qestion-but'} type='button' value='Page2' onClick={Page2Form}>質問例</button>
                </div>
            </div>
        </div>
    );
};

export default Home;