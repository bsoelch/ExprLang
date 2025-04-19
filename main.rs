use std::fs;
use std::io::{self, Write};

// TODO figure out how to do multi-file projects
// tokenizer
#[derive(Debug,PartialEq)]
enum TokenType{
    Identifier,
    Keyword,
    Number,
    Operator
}
impl ToString for TokenType {
    fn to_string(&self) -> String {
        match self {
            TokenType::Identifier => "Identifier",
            TokenType::Keyword => "Keyword",
            TokenType::Number => "Number",
            TokenType::Operator => "Operator",
        }.to_string()
    }
}
fn is_keyword(token_value: &str) -> bool {
    match token_value {
        "if" | "else" | "and" | "or" | "for" | "in" | "return" => true,
        _ => false
    }
}
const OPERATOR_CHARS: &str = "+-*/%&|^<>=!~{}()[],;:$@";
fn is_special_char(op_char: char) -> bool {
    OPERATOR_CHARS.contains(op_char)
}
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

#[derive(Debug)]
struct Position{
    line :i32,
    line_pos: i32
}
#[derive(Debug)]
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
    return output
}

// parser
#[derive(Debug,PartialEq)]
enum NodeType<'a> {
    Program,
    Assignment,
    Declaration,
    Identifier(&'a str),
    IdentifierList,
    If,
    IfElse,
    Function,
    Expression,
}
impl<'a> ToString for NodeType<'a> {
    fn to_string(&self) -> String {
        match self {
            NodeType::Program => "Program".to_string(),
            NodeType::Assignment => "Assignment".to_string(),
            NodeType::Declaration => "Declaration".to_string(),
            NodeType::Identifier(name) => format!("Identifier: \"{}\"",name),
            NodeType::IdentifierList => "IdentifierList".to_string(),
            NodeType::Function => "Function".to_string(),
            NodeType::If => "If".to_string(),
            NodeType::IfElse => "IfElse".to_string(),
            NodeType::Expression => "Expression".to_string(),
        }
    }
}
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
    let mut node_type = NodeType::Expression;
    let mut id_list = None;
    let mut consumed = 0;
    // is declaration or assignemnt
    match try_parse_identifier_list(tokens) {
      Ok((id,offset)) => {
        if offset+1 < tokens.len() && tokens[offset].token_type == TokenType::Operator &&
            (tokens[offset].value == "=" || tokens[offset].value == ":=") {
            node_type = if tokens[offset].value == "=" { NodeType::Assignment } else { NodeType::Declaration };
            id_list = Some(id);
            tokens = &tokens[(offset+1)..];
            consumed = offset+1;
        }
      }
      Err(_)=> {} // ignore
    }
    let (expr,k) = try_parse_expression(tokens)?;
    consumed += k;
    // optional semi-colon
    if k < tokens.len() && tokens[k].token_type == TokenType::Operator && tokens[k].value == ";" {
       consumed += 1;
    }
    match node_type {
        NodeType::Expression => Ok((expr,consumed)),
        NodeType::Assignment | NodeType::Declaration => {
            Ok((Node{node_type: node_type, children: vec![id_list.unwrap(),expr]},consumed))
        },
        _ => unreachable!()
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
fn is_binary_operator<'a>(token: &Token<'a>) -> bool {
    // TODO? conbine check for operator and computation of precedence
    return false
}
fn try_parse_expression<'a>(tokens: &'a [Token<'a>]) -> Result<(Node<'a>,usize),&'a Token<'a>> {
    let (expr,offset) = try_parse_operand(tokens)?;
    // TODO! resolve binary operators (using precedence climbining algo?)
    return Ok((expr,offset))
}
fn try_parse_operand<'a>(tokens: &'a [Token<'a>]) -> Result<(Node<'a>,usize),&'a Token<'a>> {
    // if-else
    if tokens[0].token_type == TokenType::Keyword && tokens[0].value == "if" {
        let mut offset = 1;
        let (condition,cond_size) = try_parse_expression(&tokens[offset..])?;
        offset += cond_size;
        let (if_body,if_size) = try_parse_expression(&tokens[offset..])?;
        if tokens[offset].token_type == TokenType::Keyword && tokens[offset].value == "else" {
            offset+=if_size + 1;
            let (else_body,else_size) = try_parse_expression(&tokens[offset..])?;
            return Ok((Node{node_type:NodeType::IfElse,children: vec![condition,if_body,else_body]},offset+else_size));
        }
        return Ok((Node{node_type:NodeType::If,children: vec![condition,if_body]},offset+if_size));
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
    // unary-operator
    // TODO unary operators
    // primitive
    if tokens[0].token_type == TokenType::Identifier {
        return Ok((Node{node_type: NodeType::Identifier(tokens[0].value), children: Vec::new()},1));
    }
    Err(&tokens[0])
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
