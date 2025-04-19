use std::fs;
use std::io::{self, Write};
use std::collections::HashMap;
use std::sync::OnceLock;

// TODO figure out how to do multi-file projects
// tokenizer
#[derive(Debug,PartialEq,Clone,Copy)]
enum TokenType{
    Identifier,
    Keyword,
    Number,
    Operator,
    EOF
}
impl ToString for TokenType {
    fn to_string(&self) -> String {
        match self {
            TokenType::Identifier => "Identifier",
            TokenType::Keyword => "Keyword",
            TokenType::Number => "Number",
            TokenType::Operator => "Operator",
            TokenType::EOF => "EOF",
        }.to_string()
    }
}
fn is_keyword(token_value: &str) -> bool {
    match token_value {
        "if" | "else" | "and" | "or" | "for" | "in" | "return" => true,
        _ => false
    }
}
const OPERATOR_CHARS: &str = "+-*/%&|^<>=!~{}()[],;:$@?";
fn is_special_char(op_char: char) -> bool {
    OPERATOR_CHARS.contains(op_char)
}
// TODO! ensure correct multi-char operators are matched
fn continue_operator(prefix: &str, suffix: char) -> bool {
    match suffix {
        '=' => match prefix {
            "+" | "-" | "*" | "/" | "%" | "&" | "|" | "^" | "=" | "!" | "<" | ">" | ":" => true,
            _ => false
        }
        '>' =>  match prefix {
            "=" | "-" | ">" | ">>" => true,
            _ => false
        }
        '<' =>  match prefix {
            "<" => true,
            _ => false
        }
        _ => false
    }
}

#[derive(Debug,Clone,Copy)]
struct Position{
    line :i32,
    line_pos: i32
}
#[derive(Debug,Clone,Copy)]
struct Token<'a>{
    token_type: TokenType,
    value: &'a str,
    pos: Position
}

impl ToString for Token<'_> {
    fn to_string(&self) -> String {
        format!("{:?}: \"{}\" at {}:{}", self.token_type, self.value, self.pos.line, self.pos.line_pos)
    }
}

fn tokenize<'a>(input: &'a str) -> Vec<Token<'a>> {
    // TODO support strings and comments
    let mut start_index: usize=0;
    let mut line = 1;
    let mut line_pos = 0;
    let mut output = Vec::new();
    let mut in_operator = false;
    for (i, c) in input.chars().enumerate() {
        if c == '\n' {
            line_pos = 0;
            line += 1;
        } else {
            line_pos += 1;
        }
        let is_operator = is_special_char(c);
        if c.is_whitespace() || if in_operator {!continue_operator(&input[start_index..i],c)} else {is_operator} {
            if start_index < i {
                let token_value = &input[start_index..i];
                let first_char = token_value.chars().next().unwrap();
                let token_type = if first_char.is_digit(10) {
                    TokenType::Number
                } else if is_special_char(first_char) {
                    TokenType::Operator
                }else if is_keyword(token_value) {
                    TokenType::Keyword
                } else {
                    TokenType::Identifier
                };
                // TODO use position at start of token instead of end
                output.push(Token{
                    token_type: token_type,
                    value: token_value,
                    pos: Position{line:line,line_pos: line_pos}
                });
            }
            start_index= if c.is_whitespace() { i+1 } else { i };
            in_operator = is_operator
        }
    }
    // ensure last token is finished
    let token_value = &input[start_index..];
    if token_value.len() > 0 {
        // TODO? create function for resolving of token-type
        let first_char = token_value.chars().next().unwrap();
        let token_type = if first_char.is_digit(10) {
            TokenType::Number
        } else if is_special_char(first_char) {
            TokenType::Operator
        }else if is_keyword(token_value) {
            TokenType::Keyword
        } else {
            TokenType::Identifier
        };
        output.push(Token{
            token_type: token_type,
            value: "",
            pos: Position{line:line,line_pos: line_pos}
        });
    }
    output.push(Token{
        token_type: TokenType::EOF,
        value: token_value,
        pos: Position{line:line,line_pos: line_pos}
    });
    return output
}

// parser

#[derive(Debug,PartialEq,Clone,Copy)]
enum OperatorType {
    // binary
    Multiply,
    Divide,
    Modulo,
    Add,
    Subtract,
    BitAnd,
    BitOr,
    BitXor,
    LShift,
    ARShift,
    LRShift,
    Less,
    LessOrEqual,
    Equal,
    NotEqual,
    GreaterOrEqual,
    Greater,
    Comma,
    Assign,
    Declare,
    AssignAdd,
    AssignSub,
    AssignMul,
    AssignDiv,
    AssignMod,
    AssignBitAnd,
    AssignBitOr,
    AssignBitXor,
    AssignLShift,
    AssignLRShift,
    AssignARShift,
    And,
    Or,
    // unary left
    Negate,
    Not,
    BitNot,
    Address,
    // unary right
    NonZero,
}
impl ToString for OperatorType {
    fn to_string(&self) -> String {
        match self {
            // binary
            OperatorType::Multiply => "Multiply",
            OperatorType::Divide => "Divide",
            OperatorType::Modulo => "Modulo",
            OperatorType::Add => "Add",
            OperatorType::Subtract => "Subtract",
            OperatorType::BitAnd => "BitAnd",
            OperatorType::BitOr => "BitOr",
            OperatorType::BitXor => "BitXor",
            OperatorType::LShift => "LShift",
            OperatorType::ARShift => "ARShift",
            OperatorType::LRShift => "LRShift",
            OperatorType::Less => "Less",
            OperatorType::LessOrEqual => "LessOrEqual",
            OperatorType::Equal => "Equal",
            OperatorType::NotEqual => "NotEqual",
            OperatorType::GreaterOrEqual => "GreaterOrEqual",
            OperatorType::Greater => "Greater",
            OperatorType::Comma => "Comma",
            OperatorType::Assign => "Assign",
            OperatorType::Declare => "Declare",
            OperatorType::AssignAdd => "AssignAdd",
            OperatorType::AssignSub => "AssignSub",
            OperatorType::AssignMul => "AssignMul",
            OperatorType::AssignDiv => "AssignDiv",
            OperatorType::AssignMod => "AssignMod",
            OperatorType::AssignBitAnd => "AssignBitAnd",
            OperatorType::AssignBitOr => "AssignBitOr",
            OperatorType::AssignBitXor => "AssignBitXor",
            OperatorType::AssignLShift => "AssignLShift",
            OperatorType::AssignLRShift => "AssignLRShift",
            OperatorType::AssignARShift => "AssignARShift",
            OperatorType::And => "And",
            OperatorType::Or => "Or",
            // unary left
            OperatorType::Negate => "Negate",
            OperatorType::Not => "Not",
            OperatorType::BitNot => "BitNot",
            OperatorType::Address => "Address",
            // unary right
            OperatorType::NonZero => "NonZero",
      }.to_string()
  }
}
struct OperatorInfo {
    op_type: OperatorType,
    precedence: i16,
    right_associative: bool
}
const BINARY_OPERATORS: [(&str,OperatorInfo); 33] = [
    ("*", OperatorInfo{op_type: OperatorType::Multiply,precedence: 0x70,right_associative: false}),
    ("/", OperatorInfo{op_type: OperatorType::Divide,precedence: 0x70,right_associative: false}),
    ("%", OperatorInfo{op_type: OperatorType::Modulo,precedence: 0x70,right_associative: false}),
    ("+", OperatorInfo{op_type: OperatorType::Add,precedence: 0x60,right_associative: false}),
    ("-", OperatorInfo{op_type: OperatorType::Subtract,precedence: 0x60,right_associative: false}),
    ("&", OperatorInfo{op_type: OperatorType::BitAnd,precedence: 0x52,right_associative: false}),
    ("^", OperatorInfo{op_type: OperatorType::BitXor,precedence: 0x51,right_associative: false}),
    ("|", OperatorInfo{op_type: OperatorType::BitOr,precedence: 0x50,right_associative: false}),
    ("<<", OperatorInfo{op_type: OperatorType::LShift,precedence: 0x40,right_associative: false}),
    (">>", OperatorInfo{op_type: OperatorType::ARShift,precedence: 0x40,right_associative: false}),
    (">>>", OperatorInfo{op_type: OperatorType::LRShift,precedence: 0x40,right_associative: false}),
    ("<", OperatorInfo{op_type: OperatorType::Less,precedence: 0x30,right_associative: false}),
    ("<=", OperatorInfo{op_type: OperatorType::LessOrEqual,precedence: 0x30,right_associative: false}),
    ("==", OperatorInfo{op_type: OperatorType::Equal,precedence: 0x30,right_associative: false}),
    ("!=", OperatorInfo{op_type: OperatorType::NotEqual,precedence: 0x30,right_associative: false}),
    (">=", OperatorInfo{op_type: OperatorType::GreaterOrEqual,precedence: 0x30,right_associative: false}),
    (">", OperatorInfo{op_type: OperatorType::Greater,precedence: 0x30,right_associative: false}),
    (",", OperatorInfo{op_type: OperatorType::Comma,precedence: 0x20,right_associative: false}),
    ("=", OperatorInfo{op_type: OperatorType::Assign,precedence: 0x10,right_associative: true}),
    (":=", OperatorInfo{op_type: OperatorType::Declare,precedence: 0x10,right_associative: true}),
    ("*=", OperatorInfo{op_type: OperatorType::AssignMul,precedence: 0x10,right_associative: true}),
    ("/=", OperatorInfo{op_type: OperatorType::AssignDiv,precedence: 0x10,right_associative: true}),
    ("%=", OperatorInfo{op_type: OperatorType::AssignMod,precedence: 0x10,right_associative: true}),
    ("+=", OperatorInfo{op_type: OperatorType::AssignAdd,precedence: 0x10,right_associative: true}),
    ("-=", OperatorInfo{op_type: OperatorType::AssignSub,precedence: 0x10,right_associative: true}),
    ("&=", OperatorInfo{op_type: OperatorType::AssignBitAnd,precedence: 0x10,right_associative: true}),
    ("|=", OperatorInfo{op_type: OperatorType::AssignBitOr,precedence: 0x10,right_associative: true}),
    ("^=", OperatorInfo{op_type: OperatorType::AssignBitXor,precedence: 0x10,right_associative: true}),
    ("<<=", OperatorInfo{op_type: OperatorType::AssignLShift,precedence: 0x10,right_associative: true}),
    (">>=", OperatorInfo{op_type: OperatorType::AssignARShift,precedence: 0x10,right_associative: true}),
    (">>>=", OperatorInfo{op_type: OperatorType::AssignLRShift,precedence: 0x10,right_associative: true}),
    ("and", OperatorInfo{op_type: OperatorType::And,precedence: 0x01,right_associative: false}),
    ("or", OperatorInfo{op_type: OperatorType::Or,precedence: 0x00,right_associative: false}),
];
static BINARY_OPERATOR_INFO: OnceLock<HashMap<&str,OperatorInfo>> = OnceLock::new();
fn binary_operator_info<'a>(token: &Token<'a>) -> Option<&'static OperatorInfo> {
    BINARY_OPERATOR_INFO.get_or_init(|| {
        let mut map = HashMap::new();
        for (key, value) in BINARY_OPERATORS {
            map.insert(key, value);
        }
        map
    }).get(token.value)
}
fn left_unary_operator_type<'a>(token: &Token<'a>) -> Option<OperatorType> {
    if token.token_type != TokenType::Operator {
        return None
    }
    match token.value {
        "-" => Some(OperatorType::Negate),
        "!" => Some(OperatorType::Not),
        "~" => Some(OperatorType::BitNot),
        "@" => Some(OperatorType::Address),
        _ => None
    }
}
fn right_unary_operator_type<'a>(token: &Token<'a>) -> Option<OperatorType> {
    if token.token_type != TokenType::Operator {
        return None
    }
    match token.value {
        "?" => Some(OperatorType::NonZero),
        _ => None
    }
}

#[derive(Debug,PartialEq,Clone,Copy)]
enum NodeType<'a> {
    Program,
    Identifier(&'a str),
    IdentifierList,
    If,
    IfElse,
    Function,
    BinaryOperator(OperatorType),
    UnaryOperator(OperatorType), // TODO? seperate types for left and right operators
    Number(i64),
    Scope,
    Call,
    ArrayAccess,
    Return
}
impl<'a> ToString for NodeType<'a> {
    fn to_string(&self) -> String {
        match self {
            NodeType::Program => "Program".to_string(),
            NodeType::Identifier(name) => format!("Identifier: \"{}\"",name),
            NodeType::IdentifierList => "IdentifierList".to_string(),
            NodeType::Function => "Function".to_string(),
            NodeType::If => "If".to_string(),
            NodeType::IfElse => "IfElse".to_string(),
            NodeType::Call =>  "Call".to_string(),
            NodeType::ArrayAccess =>  "ArrayAccess".to_string(),
            NodeType::BinaryOperator(op_type) => format!("BinaryOperator {}",op_type.to_string()),
            NodeType::UnaryOperator(op_type) => format!("UnaryOperator {}",op_type.to_string()),
            NodeType::Number(value) =>  format!("Number {}",value),
            NodeType::Scope =>  "Scope".to_string(),
            NodeType::Return =>  "Return".to_string(),
        }
    }
}
// TODO merge Node and NodeType, only use vec if child number is dynamic and larger than fixed bound
struct Node<'a> {
    node_type: NodeType<'a>,
    children: Vec<Node<'a> >,
}
fn dump_ast<'a>(out_file: &mut fs::File, root: &Node<'a>,indent: usize)-> io::Result<()> {
    writeln!(out_file,"{}{}{}","  ".repeat(indent),root.node_type.to_string(),if root.children.len()>0 {":"}else{""})?;
    for child in root.children.iter(){
        dump_ast(out_file,child,indent+1)?;
    }
    Ok(())
}

fn parse_program<'a>(mut tokens: &'a [Token<'a>]) -> Node<'a> {
    let mut children: Vec<Node> = Vec::new();
    while tokens.len() > 0 {
        if tokens[0].token_type == TokenType::EOF {
            break // reached end of file
        }
        match try_parse_statement(tokens) {
            Ok((expr,k)) => {
                children.push(expr);
                tokens=&tokens[k..];
            },
            Err(token) => {
                println!("Unexpected token: {:?}",token);
                break
            }
        }
    }
    Node{node_type: NodeType::Program, children: children}
}
fn try_parse_statement<'a>(mut tokens: &'a [Token<'a>]) -> Result<(Node<'a>,usize),&'a Token<'a>> {
    // TODO support return
    let mut consumed = 0;
    let is_return = tokens[0].token_type == TokenType::Keyword && tokens[0].value == "return";
    if is_return {
        consumed = 1;
        tokens = &tokens[1..];
    }
    let (expr,k) = try_parse_expression(tokens)?;
    consumed += k;
    // optional semi-colon
    if k < tokens.len() && tokens[k].token_type == TokenType::Operator && tokens[k].value == ";" {
       consumed += 1;
    }
    if is_return {
        Ok((Node{node_type:NodeType::Return,children: vec![expr]},consumed))
    } else {
        Ok((expr,consumed))
    }
}
fn try_parse_identifier_list<'a>(mut tokens: &'a [Token<'a>]) -> Result<(Node<'a>,usize),&'a Token<'a>> {
    let mut consumed = 0;
    let mut has_paren = false;
    let mut children = Vec::new();
    if tokens[0].token_type == TokenType::Operator && tokens[0].value == "(" {
        has_paren = true;
        consumed+=1;
        tokens=&tokens[1..];
    }
    loop {
        if has_paren && tokens[0].token_type == TokenType::Operator && tokens[0].value == ")" {
            return Ok((Node{node_type: NodeType::IdentifierList, children: children},consumed+1))
        }
        if consumed > if has_paren {1} else {0} {
            if tokens[0].token_type == TokenType::Operator && tokens[0].value == "," {
                consumed+=1;
                tokens=&tokens[1..];
            } else {
                return Ok((Node{node_type: NodeType::IdentifierList, children: children},consumed))
            }
        }
        if tokens[0].token_type == TokenType::Identifier {
            children.push(Node{node_type: NodeType::Identifier(tokens[0].value), children: Vec::new()});
            consumed+=1;
            tokens=&tokens[1..];
        } else {
            return Err(&tokens[0])
        }
    }
}
fn try_parse_expression<'a>(tokens: &'a [Token<'a>]) -> Result<(Node<'a>,usize),&'a Token<'a>> {
    let (lhs,offset) = try_parse_operand(tokens)?;
    let (expr,expr_size) = try_parse_expression1(lhs,&tokens[offset..],0)?;
    return Ok((expr,expr_size+offset));
}
// TODO? merge chains of `,` to single operation
fn try_parse_expression1<'a>(mut lhs: Node<'a>,mut tokens: &'a [Token<'a>], min_precedence: i16) -> Result<(Node<'a>,usize),&'a Token<'a>> {
    let mut consumed = 0;
    while tokens.len() > 0 {
        // check if next token is operator
        let mut next = &tokens[0];
        let op_info = match binary_operator_info(&next){
            Some(op_data) => op_data,
            None => return Ok((lhs,consumed))
        };
        if op_info.precedence < min_precedence {
            break
        }
        // consume operator
        consumed += 1;
        tokens = &tokens[1..];
        let (mut rhs,mut rhs_size) = try_parse_operand(tokens)?;
        consumed += rhs_size;
        tokens = &tokens[rhs_size..];
        if tokens.len() > 0 {
            // check next operator
            next = &tokens[0];
            let mut op_info0 = binary_operator_info(&next);
            while op_info0.is_some() && op_info0.unwrap().precedence >= op_info.precedence + if op_info0.unwrap().right_associative {0} else {1} {
                // consume operator
                (rhs, rhs_size) = try_parse_expression1(rhs,tokens,op_info.precedence + if op_info0.unwrap().precedence > op_info.precedence {1} else {0})?;
                consumed += rhs_size;
                tokens = &tokens[rhs_size..];
                next = &tokens[0];
                op_info0 = binary_operator_info(&next);
            }
        }
        lhs = Node{node_type:NodeType::BinaryOperator(op_info.op_type),children:vec![lhs,rhs]};
    }
    return Ok((lhs,consumed))
}
fn try_parse_operand<'a>(mut tokens: &'a [Token<'a>]) -> Result<(Node<'a>,usize),&'a Token<'a>> {
    // if-else
    if tokens[0].token_type == TokenType::Keyword && tokens[0].value == "if" {
        let mut offset = 1;
        let (condition,cond_size) = try_parse_expression(&tokens[offset..])?;
        offset += cond_size;
        let (if_body,if_size) = try_parse_expression(&tokens[offset..])?;
        offset += if_size;
        if tokens[offset].token_type == TokenType::Keyword && tokens[offset].value == "else" {
            offset+= 1;
            let (else_body,else_size) = try_parse_expression(&tokens[offset..])?;
            return Ok((Node{node_type:NodeType::IfElse,children: vec![condition,if_body,else_body]},offset+else_size));
        }
        return Ok((Node{node_type:NodeType::If,children: vec![condition,if_body]},offset));
    }
    // for
    // TODO? for-body
    // function
    let res = try_parse_identifier_list(tokens);
    match res{
      Ok((args,offset)) => {
        if tokens[offset].token_type == TokenType::Operator && tokens[offset].value == "=>" {
            let (body,body_size) = try_parse_statement(&tokens[offset+1..])?;
            return Ok((Node{node_type: NodeType::Function, children: vec![args,body]},offset+body_size+1));
        }},
      Err(_) => {}
    }
    let mut consumed = 0;
    let mut left_operators: Vec<OperatorType> = Vec::new();
    // left unary-operators
    while tokens.len() > 0 {
        let op_type = left_unary_operator_type(&tokens[0]);
        if op_type.is_none() { break; }
        left_operators.push(op_type.unwrap());
        consumed += 1;
        tokens = &tokens[1..];
    }
    let mut expr;
    // paren
    (expr, consumed) = if tokens[0].token_type == TokenType::Operator && tokens[0].value == "(" {
        let mut offset = 1;
        let (body,cond_size) = try_parse_expression(&tokens[offset..])?;
        offset += cond_size;
        if tokens[offset].token_type != TokenType::Operator || tokens[offset].value != ")" {
            return Err(&tokens[offset]);
        }
        (body, offset+1)
    // scope
    } else if tokens[0].token_type == TokenType::Operator && tokens[0].value == "{" {
        let mut offset = 1;
        let mut children = Vec::new();
        loop {
            let (body,cond_size) = try_parse_statement(&tokens[offset..])?;
            children.push(body);
            offset += cond_size;
            if tokens[offset].token_type == TokenType::Operator && tokens[offset].value == "}" {
                break (Node{node_type: NodeType::Scope,children: children},offset+1)
            }
        }
    // primitive
    } else if tokens[0].token_type == TokenType::Identifier {
        (Node{node_type: NodeType::Identifier(tokens[0].value), children: Vec::new()},1)
    } else if tokens[0].token_type == TokenType::Number {
        // TODO custom number parser
        match tokens[0].value.parse::<i64>() {
            Ok(value) => (Node{node_type: NodeType::Number(value), children: Vec::new()},1),
            // TODO? float support
            Err(_) => return Err(&tokens[0])
        }
    } else {
        return Err(&tokens[0]);
    };
    tokens = &tokens[consumed..];
    while tokens.len() > 0 {
        let op_type = right_unary_operator_type(&tokens[0]);
        if op_type.is_some() {
            expr = Node{node_type: NodeType::UnaryOperator(op_type.unwrap()), children: vec![expr]};
            consumed += 1;
            tokens = &tokens[1..];
        } else if tokens[0].token_type == TokenType::Operator && tokens[0].value == "(" {
            if tokens[1].token_type == TokenType::Operator && tokens[1].value == ")" {
                consumed += 2;
                tokens = &tokens[2..];
                expr = Node{node_type: NodeType::Call, children: vec![expr]};
            } else {
                tokens = &tokens[1..];
                let (args,arg_size) = try_parse_expression(tokens)?;
                if tokens[arg_size].token_type != TokenType::Operator || tokens[arg_size].value != ")" {
                    return Err(&tokens[arg_size]);
                }
                consumed += arg_size+2;
                tokens = &tokens[arg_size+1..];
                expr = Node{node_type: NodeType::Call, children: vec![expr,args]};
            }
        } else if tokens[0].token_type == TokenType::Operator && tokens[0].value == "[" {
            if tokens[1].token_type == TokenType::Operator && tokens[1].value == "]" {
                consumed += 2;
                tokens = &tokens[2..];
                expr = Node{node_type: NodeType::ArrayAccess, children: vec![expr]};
            } else {
                tokens = &tokens[1..];
                let (args,arg_size) = try_parse_expression(tokens)?;
                if tokens[arg_size].token_type != TokenType::Operator || tokens[arg_size].value != "]" {
                    return Err(&tokens[arg_size]);
                }
                consumed += arg_size+2;
                tokens = &tokens[arg_size+1..];
                expr = Node{node_type: NodeType::ArrayAccess, children: vec![expr,args]};
            }
        } else {
            break;
        }
    }
    // apply left unary operators to expression
    for op_type in left_operators.into_iter().rev() {
       expr = Node{node_type: NodeType::UnaryOperator(op_type), children: vec![expr]};
    }
    return Ok((expr,consumed));
}


// main

fn main() -> io::Result<()> {
    // Read the content of the input file
    let input = fs::read_to_string("in.txt")?;

    let tokens = tokenize(&input);

    let ast = parse_program(&tokens);

    // Write the output to the output file
    let mut out_file = fs::File::create("tokens.txt")?;
    for token_string in tokens.iter().map(|token| token.to_string()) {
        writeln!(out_file, "{}", token_string)?;
    }
    out_file = fs::File::create("ast.txt")?;
    dump_ast(&mut out_file,&ast, 0)?;

    Ok(())
}
