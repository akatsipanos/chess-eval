import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Chess } from 'chess.js'
import { Chessboard } from 'react-chessboard'
import axios from 'axios'
import './App.css'

function App() {
    const chessGameRef = useRef(new Chess())
    const chessGame = chessGameRef.current
    const [chessPosition, setChessPosition] = useState(chessGame.fen())
    const [fenInput, setFenInput] = useState(chessGame.fen())
    const [whiteTime, setWhiteTime] = useState('600')
    const [blackTime, setBlackTime] = useState('600')
    const [whiteRating, setWhiteRating] = useState('1800')
    const [blackRating, setBlackRating] = useState('1800')
    const [turn, setTurn] = useState('white')
    const [pred, setPred] = useState({ result: '-', probs: [0, 0, 0], sf_eval: 0 })
    const [moveFrom, setMoveFrom] = useState('')
    const [optionSquares, setOptionSquares] = useState({})

    const api = useMemo(() => axios.create({ baseURL: 'http://localhost:8000' }), [])

    const fetchPrediction = useCallback(async (fenValue) => {
        try {
            const payload = {
                fen_number: fenValue,
                white_time: whiteTime,
                black_time: blackTime,
                white_rating: whiteRating,
                black_rating: blackRating,
                turn,
            }
            const { data } = await api.post('/api/predict', payload)
            setPred(data)
        } catch (err) {
            console.error('Prediction error:', err)
        }
    }, [api, whiteTime, blackTime, whiteRating, blackRating, turn])

    useEffect(() => {
        fetchPrediction(chessGame.fen())
    }, [])

    function getMoveOptions(square) {
        const moves = chessGame.moves({
            square,
            verbose: true
        })

        if (moves.length === 0) {
            setOptionSquares({})
            return false
        }

        const newSquares = {}

        for (const move of moves) {
            newSquares[move.to] = {
                background: chessGame.get(move.to) && chessGame.get(move.to)?.color !== chessGame.get(square)?.color
                    ? 'radial-gradient(circle, rgba(0,0,0,.1) 85%, transparent 85%)'
                    : 'radial-gradient(circle, rgba(0,0,0,.1) 25%, transparent 25%)',
                borderRadius: '50%'
            }
        }

        newSquares[square] = {
            background: 'rgba(255, 255, 0, 0.4)'
        }

        setOptionSquares(newSquares)
        return true
    }

    function onSquareClick({ square, piece }) {
        // piece clicked to move
        if (!moveFrom && piece) {
            const hasMoveOptions = getMoveOptions(square)
            if (hasMoveOptions) {
                setMoveFrom(square)
            }
            return
        }

        // square clicked to move to, check if valid move
        const moves = chessGame.moves({
            square: moveFrom,
            verbose: true
        })
        const foundMove = moves.find(m => m.from === moveFrom && m.to === square)

        // not a valid move
        if (!foundMove) {
            const hasMoveOptions = getMoveOptions(square)
            setMoveFrom(hasMoveOptions ? square : '')
            return
        }

        // is normal move
        try {
            chessGame.move({
                from: moveFrom,
                to: square,
                promotion: 'q'
            })
        } catch {
            const hasMoveOptions = getMoveOptions(square)
            if (hasMoveOptions) {
                setMoveFrom(square)
            }
            return
        }

        const newFen = chessGame.fen()
        setChessPosition(newFen)
        setFenInput(newFen)
        setTurn(chessGame.turn() === 'w' ? 'white' : 'black')
        setMoveFrom('')
        setOptionSquares({})
        fetchPrediction(newFen)
    }

    function onPieceDrop({ sourceSquare, targetSquare }) {
        if (!targetSquare) {
            return false
        }

        try {
            chessGame.move({
                from: sourceSquare,
                to: targetSquare,
                promotion: 'q'
            })

            const newFen = chessGame.fen()
            setChessPosition(newFen)
            setFenInput(newFen)
            setTurn(chessGame.turn() === 'w' ? 'white' : 'black')
            setMoveFrom('')
            setOptionSquares({})
            fetchPrediction(newFen)

            return true
        } catch {
            return false
        }
    }

    const onFenSubmit = useCallback((e) => {
        e.preventDefault()
        try {
            const trimmed = fenInput.trim()
            const normalized = trimmed.includes(' ') ? trimmed : `${trimmed} w - - 0 1`

            const newGame = new Chess(normalized)
            chessGameRef.current = newGame

            const f = newGame.fen()
            setChessPosition(f)
            setFenInput(f)
            setTurn(newGame.turn() === 'w' ? 'white' : 'black')
            setMoveFrom('')
            setOptionSquares({})
            fetchPrediction(f)
        } catch (err) {
            console.error('FEN error:', err)
            alert('Invalid FEN')
        }
    }, [fenInput, fetchPrediction])

    const evalPercent = useMemo(() => {
        const v = Math.max(-5, Math.min(5, pred.sf_eval))
        return Math.round(((v + 5) / 10) * 100)
    }, [pred.sf_eval])

    const chessboardOptions = {
        position: chessPosition,
        onPieceDrop,
        onSquareClick,
        squareStyles: optionSquares,
        id: 'chess-eval-board',
        boardWidth: 400,
        arePiecesDraggable: true,
        customBoardStyle: {
            borderRadius: '4px',
            boxShadow: '0 2px 10px rgba(0, 0, 0, 0.5)',
        }
    }

    return (
        <div className="app-container">
            <div className="chessboard-section">
                <Chessboard options={chessboardOptions} />

                <div className="chessboard-info">
                    {moveFrom && <div className="selected-square">Selected: {moveFrom}</div>}
                    <div className="fen-display">FEN: {chessPosition.substring(0, 40)}...</div>
                </div>
            </div>

            <div className="controls-section">
                <h2>Chess Eval</h2>

                <div className="input-form">
                    <div className="form-group">
                        <label>FEN</label>
                        <input value={fenInput} onChange={(e) => setFenInput(e.target.value)} />
                    </div>
                    <div className="input-row">
                        <div className="form-group">
                            <label>White time (s)</label>
                            <input value={whiteTime} onChange={(e) => setWhiteTime(e.target.value)} />
                        </div>
                        <div className="form-group">
                            <label>Black time (s)</label>
                            <input value={blackTime} onChange={(e) => setBlackTime(e.target.value)} />
                        </div>
                    </div>
                    <div className="input-row">
                        <div className="form-group">
                            <label>White rating</label>
                            <input value={whiteRating} onChange={(e) => setWhiteRating(e.target.value)} />
                        </div>
                        <div className="form-group">
                            <label>Black rating</label>
                            <input value={blackRating} onChange={(e) => setBlackRating(e.target.value)} />
                        </div>
                    </div>
                    <div className="form-group">
                        <label>Turn</label>
                        <select value={turn} onChange={(e) => setTurn(e.target.value)}>
                            <option value="white">White</option>
                            <option value="black">Black</option>
                        </select>
                    </div>
                    <button type="button" className="submit-button" onClick={onFenSubmit}>
                        Apply / Evaluate
                    </button>
                </div>

                <div className="evaluation-section">
                    <h3>Evaluation</h3>
                    <div className="evaluation-bar-container">
                        <div className="evaluation-bar">
                            <div className="evaluation-bar-fill" style={{ height: `${evalPercent}%` }} />
                        </div>
                        <div className="evaluation-labels">
                            <div className="eval-result"><strong>Result:</strong> {pred.result}</div>
                            <div className="eval-probs">
                                <strong>Probs:</strong> W {Math.round(pred.probs[0] * 100)}% | D {Math.round(pred.probs[1] * 100)}% | B {Math.round(pred.probs[2] * 100)}%
                            </div>
                            <div className="eval-sf"><strong>SF eval:</strong> {pred.sf_eval}</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default App
