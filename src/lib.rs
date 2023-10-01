use pyo3::prelude::*;
use llm::{Model, LoadProgress};
use std::io::{Write, Read};
use std::fs::File;
use chrono::{DateTime, Local};
use base64::{Engine, engine::GeneralPurpose, engine::GeneralPurposeConfig, alphabet::STANDARD};
use serde::{Deserialize, Serialize};
mod database;
use database::{Database, Message, CompanionData, UserData};
mod vectordb;
use vectordb::VectorDatabase;


#[pyfunction]
fn init_default() -> PyResult<()> {
    match Database::create() {
        Ok(_) => { println!("Successfully connected to local database"); }
        Err(e) => { eprintln!("Cannot connect to SQLite database because of: {}",e); }
    }

    match VectorDatabase::connect() {
        Ok(_) => { println!("Successfully connected to tantivy"); }
        Err(e) => { eprintln!("Cannot connect to tantivy because of: {}",e); }
    }
    Ok(())
}

#[pyfunction]
fn init(companion_name: String, companion_persona: String, example_dialogue: String, first_message: String,
        long_term_memory_limit: u32, short_term_memory_limit: u32, roleplay: bool, username: String, user_persona: String) -> PyResult<()> {
    match Database::create() {
        Ok(_) => { println!("Successfully connected to local database"); }
        Err(e) => { eprintln!("Cannot connect to SQLite database because of: {}",e); }
    }

    match VectorDatabase::connect() {
        Ok(_) => { println!("Successfully connected to tantivy"); }
        Err(e) => { eprintln!("Cannot connect to tantivy because of: {}",e); }
    }

    match Database::change_companion(&companion_name, &companion_persona, &example_dialogue, &first_message, long_term_memory_limit, short_term_memory_limit, roleplay) {
        Ok(_) => {},
        Err(e) => {
            eprintln!("Error while changing companion data in sqlite database: {}", e);
        },
    }

    match Database::change_user(&username, &user_persona) {
        Ok(_) => {},
        Err(e) => {
            eprintln!("Error while changing user data in sqlite database: {}", e);
        },
    }
    Ok(())
}

fn load_progress_callback(_: LoadProgress) {}

#[pyfunction]
fn prompt(text: String, model_path: String) -> PyResult<String> {
    match Database::add_message(&text, false) {
        Ok(_) => {},
        Err(e) => {
            eprintln!("Error while adding message to database/short-term memory: {}", e);
        },
    };
    let vector = match VectorDatabase::connect() {
        Ok(vd) => vd,
        Err(e) => {
            eprintln!("Error while connecting to tantivy: {}", e);
            panic!();
        }
    };
    let local: DateTime<Local> = Local::now();
    let formatted_date = local.format("* at %A %d.%m.%Y %H:%M *\n").to_string();
    let mut is_llama2: bool = false;

    // https://github.com/rustformers/llm
    // https://docs.rs/llm/latest/llm/

    if model_path.ends_with(".bin") {
        is_llama2 = model_path.contains("llama");
    }
    if model_path.is_empty() {
        eprintln!("Incorrect ai model path");
        panic!();
    }

    let llama = llm::load::<llm::models::Llama>(
        std::path::Path::new(&model_path),
        llm::TokenizerSource::Embedded,
        llm::ModelParameters::default(),
        load_progress_callback
    )
    .unwrap_or_else(|err| panic!("Failed to load model: {err}"));
    
    let mut session = llama.start_session(Default::default());
    let x: String;
    println!("Generating ai response...");
    let companion: CompanionData = match Database::get_companion_data() {
        Ok(cd) => cd,
        Err(e) => {
            eprintln!("Error while getting companion data from sqlite database: {}", e);
            panic!();
        }
    };
    let user: UserData = match Database::get_user_data() {
        Ok(ud) => ud,
        Err(e) => {
            eprintln!("Error while getting user data from sqlite database: {}", e);
            panic!();
        }
    };
    let mut base_prompt: String;
    let mut rp: &str = "";
    if companion.roleplay == 1 {
        rp = "gestures and other non-verbal actions are written between asterisks (for example, *waves hello* or *moves closer*)";
    }
    if is_llama2 {
        base_prompt = 
        format!("<<SYS>>\nYou are {}, {}\nyou are talking with {}, {} is {}\n{}\n[INST]\n{}\n[/INST]",
                companion.name, companion.persona.replace("{{char}}", &companion.name).replace("{{user}}", &user.name), user.name, user.name, user.persona.replace("{{char}}", &companion.name).replace("{{user}}", &user.name), rp, companion.example_dialogue.replace("{{char}}", &companion.name).replace("{{user}}", &user.name));
    } else {
        base_prompt = 
        format!("Text transcript of a conversation between {} and {}. {}\n{}'s Persona: {}\n{}'s Persona: {}\n<START>{}\n<START>\n", 
                                            user.name, companion.name, rp, user.name, user.persona.replace("{{char}}", &companion.name).replace("{{user}}", &user.name), companion.name, companion.persona.replace("{{char}}", &companion.name).replace("{{user}}", &user.name), companion.example_dialogue.replace("{{char}}", &companion.name).replace("{{user}}", &user.name));
    }
    let abstract_memory: Vec<String> = match vector.get_matches(&text, companion.long_term_mem) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error while getting messages from long-term memory: {}", e);
            panic!();
        }
    };
    for message in abstract_memory {
        base_prompt += &message.replace("{{char}}", &companion.name).replace("{{user}}", &user.name);
    }
    let ai_memory: Vec<Message> = match Database::get_x_msgs(companion.short_term_mem) {
        Ok(msgs) => msgs,
        Err(e) => {
            eprintln!("Error while getting messages from database/short-term memory: {}", e);
            panic!();
        }
    };
    if is_llama2 {
        for message in ai_memory {
            let prefix = if message.ai == "true" { &companion.name } else { &user.name };
            let text = message.text;
            let formatted_message = format!("{}: {}\n", prefix, text);
            base_prompt += &("[INST]".to_owned() + &formatted_message + "[/INST]\n");
        }
        base_prompt += "<</SYS>>";
    } else {
        for message in ai_memory {
            let prefix = if message.ai == "true" { &companion.name } else { &user.name };
            let text = message.text;
            let formatted_message = format!("{}: {}\n", prefix, text);
            base_prompt += &formatted_message;
        }
    }
    let mut endOfGeneration = String::new();
    let eog = format!("\n{}:", user.name);
    let res = session.infer::<std::convert::Infallible>(
        &llama,
        &mut rand::thread_rng(),
        &llm::InferenceRequest {
            prompt: llm::Prompt::Text(&format!("{}{}:", &base_prompt, companion.name)),
            parameters: &llm::InferenceParameters::default(),
            play_back_previous_tokens: false,
            maximum_token_count: None,
        },
        &mut Default::default(),
        |t| {
            match t {
                llm::InferenceResponse::SnapshotToken(_) => {/*print!("{token}");*/}
                llm::InferenceResponse::PromptToken(_) => {/*print!("{token}");*/}
                llm::InferenceResponse::InferredToken(token) => {
                    //x = x.clone()+&token;
                    endOfGeneration.push_str(&token);
                    print!("{token}");
                    if endOfGeneration.contains(&eog) {
                        return Ok(llm::InferenceFeedback::Halt);          
                    }
                }
                llm::InferenceResponse::EotToken => {}
            }
            std::io::stdout().flush().unwrap();
            Ok(llm::InferenceFeedback::Continue)
        }
    );
    x = endOfGeneration.replace(&eog, "");
    match res {
        Ok(result) => println!("\n\nInference stats:\n{result}"),
        Err(err) => println!("\n{err}"),
    }
    let companion_text = x
    .split(&format!("\n{}: ", &companion.name))
    .next()
    .unwrap_or("");
    match Database::add_message(companion_text, true) {
        Ok(_) => {},
        Err(e) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while adding message to database/short-term memory: {:?}", e)));
        },
    };
    match vector.add_entry(&format!("{}{}: {}\n{}: {}\n", formatted_date, "{{user}}", text, "{{char}}", &companion_text)) {
        Ok(_) => {},
        Err(e) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while adding message to long-term memory: {:?}", e)));
        },
    };
    Ok(companion_text.to_string())
}

#[pyfunction]
fn get_messages() -> PyResult<Vec<Message>> {
    let messages: Vec<Message> = match Database::get_messages() {
        Ok(msgs) => msgs,
        Err(e) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while getting messages from sqlite database: {:?}", e)));
        },
    };
    Ok(messages)
}

#[pyfunction]
fn clear_messages() -> PyResult<()> {
    match Database::clear_messages() {
        Ok(_) => {},
        Err(e) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while removing messages from sqlite database: {:?}", e)));
        },
    };
    Ok(())
}

#[pyfunction]
fn rm_message(message_id: u32) -> PyResult<()> {
    match Database::rm_message(message_id) {
        Ok(_) => {},
        Err(e) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while removing message from sqlite database: {:?}", e)));
        },
    };
    Ok(())
}

#[pyfunction]
fn fetch_companion_data() -> PyResult<CompanionData> {
    let companionData: CompanionData;
    match Database::get_companion_data() {
        Ok(companion_data) => {
            companionData = companion_data
        },
        Err(e) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while getting companion data from sqlite database: {:?}", e)));
        },
    };
    Ok(companionData)
}

#[pyfunction]
fn fetch_user_data() -> PyResult<UserData> {
    let userData: UserData;
    match Database::get_user_data() {
        Ok(user_data) => {
            userData = user_data
        },
        Err(e) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while getting user data from sqlite database: {:?}", e)));
        },
    };
    Ok(userData)
}

#[pyfunction]
fn change_first_message(new_first_message: String) -> PyResult<()> {
    match Database::change_first_message(&new_first_message) {
        Ok(_) => {},
        Err(e) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while changing companion's first message in sqlite database: {:?}", e)));
        },
    };
    Ok(())
}

#[pyfunction]
fn change_companion_name(new_companion_name: String) -> PyResult<()> {
    match Database::change_companion_name(&new_companion_name) {
        Ok(_) => {},
        Err(e) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while changing companion name in sqlite database: {:?}", e)));
        },
    };
    Ok(())
}

#[pyfunction]
fn change_user_name(new_user_name: String) -> PyResult<()> {
    match Database::change_username(&new_user_name) {
        Ok(_) => {},
        Err(e) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while changing username in sqlite database: {:?}", e)));
        },
    };
    Ok(())
}

#[pyfunction]
fn change_companion_persona(new_companion_persona: String) -> PyResult<()> {
    match Database::change_companion_persona(&new_companion_persona) {
        Ok(_) => {},
        Err(e) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while changing companion persona in sqlite database: {:?}", e)));
        },
    };
    Ok(())
}

#[pyfunction]
fn change_companion_example_dialogue(new_example_dialogue: String) -> PyResult<()> {
    match Database::change_companion_example_dialogue(&new_example_dialogue) {
        Ok(_) => {},
        Err(e) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while changing companion example dialogue in sqlite database: {:?}", e)));
        },
    };
    Ok(())
}

#[pyfunction]
fn change_user_persona(new_user_persona: String) -> PyResult<()> {
    match Database::change_user_persona(&new_user_persona) {
        Ok(_) => {},
        Err(e) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while changing user persona in sqlite database: {:?}", e)));
        },
    };
    Ok(())
}

#[pyfunction]
fn change_companion_data(new_companion_name: String, new_companion_persona: String, new_example_dialogue: String, new_first_message: String, long_term_memory_limit: u32, short_term_memory_limit: u32, roleplay: bool) -> PyResult<()> {
    match Database::change_companion(&new_companion_name, &new_companion_persona, &new_example_dialogue, &new_first_message, long_term_memory_limit, short_term_memory_limit, roleplay) {
        Ok(_) => {},
        Err(e) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while changing companion data in sqlite database: {:?}", e)));
        },
    }
    Ok(())
}

#[pyfunction]
fn change_user_data(new_user_name: String, new_user_persona: String) -> PyResult<()> {
    match Database::change_user(&new_user_name, &new_user_persona) {
        Ok(_) => {},
        Err(e) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while changing user data in sqlite database: {:?}", e)));
        },
    };
    Ok(())
}

#[pyfunction]
fn add_custom_data(text: String) -> PyResult<()> {
    match VectorDatabase::connect() {
        Ok(vdb) => {
            match vdb.add_entry(&(text+"\n")) {
                Ok(_) => {},
                Err(e) => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while adding custom data to long-term memory: {:?}", e)));
                },
            };
        },
        Err(e) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while adding custom data to long-term memory: {:?}", e)));
        },
    };
    Ok(())
}

#[pyfunction]
fn erase_longterm_mem() -> PyResult<()> {
    match VectorDatabase::connect() {
        Ok(vdb) => {
            match vdb.erase_memory() {
                Ok(_) => {},
                Err(e) => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while erasing data from long-term memory: {:?}", e)));
                },
            };
        },
        Err(e) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while erasing data from long-term memory: {:?}", e)));
        },
    };
    Ok(())
}

#[pyfunction]
fn change_longterm_memory_limit(new_limit: u32) -> PyResult<()> {
    match Database::change_long_term_memory(new_limit) {
        Ok(_) => {},
        Err(e) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while changing long-term memory limit in sqlite database: {:?}", e)));
        }
    };
    Ok(())
}

#[pyfunction]
fn change_shortterm_memory_limit(new_limit: u32) -> PyResult<()> {
    match Database::change_short_term_memory(new_limit) {
        Ok(_) => {},
        Err(e) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while changing short-term memory limit in sqlite database: {:?}", e)));
        },
    };
    Ok(())
}

#[pyfunction]
fn change_roleplay(enable: bool) -> PyResult<()> {
    match Database::disable_enable_roleplay(enable) {
        Ok(_) => {},
        Err(e) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while enabling/disabling roleplay in sqlite database: {:?}", e)));
        },
    };
    Ok(())
}

// works with https://zoltanai.github.io/character-editor/
#[derive(Deserialize)]
struct CharacterJson {
    name: String,
    description: String,
    first_mes: String,
    mes_example: String,
}

#[pyfunction]
fn import_character_json(character_json_text: String) -> PyResult<()> {
    let character_json: CharacterJson = match serde_json::from_str(&character_json_text) {
        Ok(v) => v,
        Err(e) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while parsing provided text as json: {:?}", e)));
        }
    };
    match Database::import_companion(&character_json.name, &character_json.description, &character_json.mes_example, &character_json.first_mes) {
        Ok(_) => Ok(()),
        Err(e) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while importing character via character class to sqlite database {:?}", e)));
        },
    }
}

#[derive(Deserialize)]
struct CharacterCard {
    name: String,
    description: String,
    first_mes: String,
    mes_example: String,
}

#[pyfunction]
fn import_character_card(character_card_path: String) -> PyResult<()> {
    let mut card_file = File::open(character_card_path).expect("File at this path does not exist");
    let mut data = Vec::new();
    match card_file.read_to_end(&mut data) {
        Ok(_) => {},
        Err(e) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while reading character card file: {:?}", e)));
        }
    };
    let text_chunk_start = data.windows(9).position(|window| window == b"tEXtchara").expect("Looks like this image does not contain character data");
    let text_chunk_end = data.windows(4).rposition(|window| window == b"IEND").expect("Looks like this image does not contain character data");
    let character_base64 = &data[text_chunk_start + 10..text_chunk_end - 8];
    let engine = GeneralPurpose::new(&STANDARD, GeneralPurposeConfig::new());
    let character_bytes = match engine.decode(character_base64) {
        Ok(b) => b,
        Err(e) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while decoding base64 character data from character card: {:?}", e)));
        }
    };
    let character_text: &str = match std::str::from_utf8(&character_bytes) {
        Ok(s) => s,
        Err(e) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while parsing decoded base64 bytes to utf8 string: {:?}", e)));
        }
    };
    let character_data: CharacterCard = serde_json::from_str(character_text).expect("Your image file does not contain correct json data");
    match Database::import_companion(&character_data.name, &character_data.description, &character_data.mes_example, &character_data.first_mes) {
        Ok(_) => {},
        Err(e) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while importing companion data via character card: {:?}", e)));
        }
    };
    Ok(())
}


#[derive(Deserialize, Serialize)]
struct MessagesJson {
    messages: Vec<MessageImport>,
}

#[derive(Deserialize, Serialize)]
struct MessageImport {
    ai: bool,
    text: String,
}

#[pyfunction]
fn import_messages_json(messages_json_text: String) -> PyResult<()> {
    let messages_json: MessagesJson = match serde_json::from_str(&messages_json_text) {
        Ok(v) => v,
        Err(e) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while parsing provided text as json: {:?}", e)));
        }
    };
    let mut messages_iter = messages_json.messages.iter();
    for message in messages_iter.to_owned() {
        match Database::add_message(&message.text, message.ai) {
            Ok(_) => {},
            Err(e) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while adding message to database/short-term memory: {:?}", e)));
            },
        };
    }
    let vector = match VectorDatabase::connect() {
        Ok(vd) => vd,
        Err(e) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while connecting to tantivy: {:?}", e)));
        }
    };
    while let Some(msg1) = messages_iter.next() {
        if let Some(msg2) = messages_iter.next() {
            match vector.add_entry(&format!("{}: {}\n{}: {}\n", if msg1.ai {"{{char}}"} else {"{{user}}"}, msg1.text, if msg2.ai {"{{char}}"} else {"{{user}}"}, msg2.text)) {
                Ok(_) => {},
                Err(e) => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while importing message to long-term memory: {:?}", e)));
                },
            };
        }
    }
    Ok(())
}

#[pyfunction]
fn get_messages_json() -> PyResult<String> {
    let database_messages = match Database::get_messages() {
        Ok(m) => m,
        Err(e) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while fetching messages from json text: {:?}", e)));
        }
    };
    let messages: MessagesJson = MessagesJson { messages: database_messages.iter().map(|message|
        MessageImport {
            ai: match message.ai.as_str() {
                "true" => true,
                "false" => false,
                _ => panic!(),
            },
            text: message.text.clone(),
        }
    ).collect(), };
    let json_messages = match serde_json::to_string(&messages) {
        Ok(v) => v,
        Err(e) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while encoding messages as json: {:?}", e)));
        },
    };
    Ok(json_messages)
}

#[pymodule]
fn ai_companion_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(init_default, m)?)?;
    m.add_function(wrap_pyfunction!(init, m)?)?;
    m.add_function(wrap_pyfunction!(prompt, m)?)?;
    m.add_function(wrap_pyfunction!(get_messages, m)?)?;
    m.add_function(wrap_pyfunction!(clear_messages, m)?)?;
    m.add_function(wrap_pyfunction!(rm_message, m)?)?;
    m.add_function(wrap_pyfunction!(fetch_companion_data, m)?)?;
    m.add_function(wrap_pyfunction!(fetch_user_data, m)?)?;
    m.add_function(wrap_pyfunction!(change_first_message, m)?)?;
    m.add_function(wrap_pyfunction!(change_companion_name, m)?)?;
    m.add_function(wrap_pyfunction!(change_user_name, m)?)?;
    m.add_function(wrap_pyfunction!(change_user_name, m)?)?;
    m.add_function(wrap_pyfunction!(change_companion_persona, m)?)?;
    m.add_function(wrap_pyfunction!(change_companion_example_dialogue, m)?)?;
    m.add_function(wrap_pyfunction!(change_user_persona, m)?)?;
    m.add_function(wrap_pyfunction!(change_companion_data, m)?)?;
    m.add_function(wrap_pyfunction!(change_user_data, m)?)?;
    m.add_function(wrap_pyfunction!(add_custom_data, m)?)?;
    m.add_function(wrap_pyfunction!(erase_longterm_mem, m)?)?;
    m.add_function(wrap_pyfunction!(change_longterm_memory_limit, m)?)?;
    m.add_function(wrap_pyfunction!(change_shortterm_memory_limit, m)?)?;
    m.add_function(wrap_pyfunction!(change_roleplay, m)?)?;
    m.add_function(wrap_pyfunction!(import_character_json, m)?)?;
    m.add_function(wrap_pyfunction!(import_character_card, m)?)?;
    m.add_function(wrap_pyfunction!(import_messages_json, m)?)?;
    m.add_function(wrap_pyfunction!(get_messages_json, m)?)?;
    Ok(())
}
