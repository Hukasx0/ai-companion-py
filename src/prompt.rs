use pyo3::prelude::*;
use llm::Model;
use llm::models::Llama;
use std::io::Write;
use chrono::{DateTime, Local};
use crate::Database;
use crate::database::{Message, CompanionData, UserData};
use crate::vectordb::VectorDatabase;

#[pyclass]
pub struct Companion {
    pub ai_model: Option<Llama>,
    pub is_llama2: bool,
}

pub fn prompt_rs(companion_py: &Companion, text_prompt: &str) -> Result<String, String> {
    let vector = match VectorDatabase::connect() {
        Ok(vd) => vd,
        Err(e) => {
            eprintln!("Error while connecting to tantivy: {}", e);
            panic!();
        }
    };
    let local: DateTime<Local> = Local::now();
    let formatted_date = local.format("* at %A %d.%m.%Y %H:%M *\n").to_string();

    let llama = companion_py.ai_model.as_ref().unwrap();
    
    let mut session = llama.start_session(Default::default());
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
    if companion_py.is_llama2 {
        base_prompt = 
        format!("<<SYS>>\nYou are {}, {}\nyou are talking with {}, {} is {}\n{}\n[INST]\n{}\n[/INST]",
                companion.name, companion.persona.replace("{{char}}", &companion.name).replace("{{user}}", &user.name), user.name, user.name, user.persona.replace("{{char}}", &companion.name).replace("{{user}}", &user.name), rp, companion.example_dialogue.replace("{{char}}", &companion.name).replace("{{user}}", &user.name));
    } else {
        base_prompt = 
        format!("Text transcript of a conversation between {} and {}. {}\n{}'s Persona: {}\n{}'s Persona: {}\n<START>{}\n<START>\n", 
                                            user.name, companion.name, rp, user.name, user.persona.replace("{{char}}", &companion.name).replace("{{user}}", &user.name), companion.name, companion.persona.replace("{{char}}", &companion.name).replace("{{user}}", &user.name), companion.example_dialogue.replace("{{char}}", &companion.name).replace("{{user}}", &user.name));
    }
    let abstract_memory: Vec<String> = match vector.get_matches(text_prompt, companion.long_term_mem) {
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
    if companion_py.is_llama2 {
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
    let mut end_of_generation = String::new();
    let eog = format!("\n{}:", user.name);
    let res = session.infer::<std::convert::Infallible>(
        llama,
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
                    end_of_generation.push_str(&token);
                    print!("{token}");
                    if end_of_generation.contains(&eog) {
                        return Ok(llm::InferenceFeedback::Halt);          
                    }
                }
                llm::InferenceResponse::EotToken => {}
            }
            std::io::stdout().flush().unwrap();
            Ok(llm::InferenceFeedback::Continue)
        }
    );
    let x: String = end_of_generation.replace(&eog, "");
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
            return Err(format!("Error while adding message to database/short-term memory: {:?}", e));
        },
    };
    match vector.add_entry(&format!("{}{}: {}\n{}: {}\n", formatted_date, "{{user}}", text_prompt, "{{char}}", &companion_text)) {
        Ok(_) => {},
        Err(e) => {
            return Err(format!("Error while adding message to long-term memory: {:?}", e));
        },
    };
    Ok(companion_text.to_string())
}
