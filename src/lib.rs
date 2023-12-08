use pyo3::prelude::*;
use llm::LoadProgress;
use std::io::Read;
use std::fs::File;
use base64::{Engine, engine::GeneralPurpose, engine::GeneralPurposeConfig, alphabet::STANDARD};
use serde::{Deserialize, Serialize};
mod database;
use database::{Database, CompanionData, UserData};
mod vectordb;
use vectordb::VectorDatabase;
mod prompt;
use prompt::{prompt_rs, Companion};

#[pymethods]
impl Companion {
    fn load_model(&mut self, ai_model_path: &str, use_gpu: bool) -> PyResult<()> {
        if ai_model_path.ends_with(".bin") {
            self.is_llama2 = ai_model_path.contains("llama");
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err("Error while loading ai model, make sure that the path to the ai model is correct, that it is a valid GGML model and that the file has a .bin extension"));
        }
        let llama = llm::load::<llm::models::Llama>(
            std::path::Path::new(ai_model_path),
            llm::TokenizerSource::Embedded,
            llm::ModelParameters {
                prefer_mmap: true,
                use_gpu: use_gpu,
                ..Default::default()
            },
            load_progress_callback
        ).unwrap_or_else(|err| panic!("Failed to load model: {err}"));
        self.ai_model = Some(llama);
        Ok(())
    }

    fn prompt(&self, text: String) -> PyResult<String> {
        match Database::add_message(&text, false) {
            Ok(_) => {},
            Err(e) => {
                eprintln!("Error while adding message to database/short-term memory: {}", e);
            },
        };
       match prompt_rs(self, &text) {
        Ok(v) => Ok(v),
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(e))
       }
    }

    fn regenerate_message(&self) -> PyResult<String> {
        match Database::remove_latest_message() {
            Ok(_) => {},
            Err(e) => {
                let error_msg = format!("Error while removing latest message from sqlite database: {}", e);
                return Err(pyo3::exceptions::PyValueError::new_err(error_msg));
            }
        }
        let previous_prompt = match Database::get_x_msgs(1) {
            Ok(v) => v,
            Err(e) => {
                let error_msg = format!("Error while fetching previous prompt from sqlite database: {}", e);
                return Err(pyo3::exceptions::PyValueError::new_err(error_msg));
            }
        };
        let previous_prompt_str = &previous_prompt[0].text;
        match prompt_rs(self, previous_prompt_str) {
            Ok(text) => Ok(text),
            Err(error) => Err(pyo3::exceptions::PyValueError::new_err(error))
        }
    }

    #[staticmethod]
    fn clear_messages() -> PyResult<()> {
        match Database::clear_messages() {
            Ok(_) => {},
            Err(e) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while removing messages from sqlite database: {:?}", e)));
            },
        };
        Ok(())
    }

    #[staticmethod]
    fn rm_message(message_id: u32) -> PyResult<()> {
        match Database::rm_message(message_id) {
            Ok(_) => {},
            Err(e) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while removing message from sqlite database: {:?}", e)));
            },
        };
        Ok(())
    }

    #[staticmethod]
    fn edit_message(new_text: &str, id: u32) -> PyResult<()> {
        match Database::modify_message(new_text, id) {
            Ok(_) => {},
            Err(e) => {
                let error_msg = format!("Error while removing message from sqlite database: {}", e);
                return Err(pyo3::exceptions::PyValueError::new_err(error_msg));
            },
        };
        Ok(())
    }

    #[staticmethod]
    fn fetch_companion_data() -> PyResult<CompanionData> {
        let companion_data: CompanionData =
        match Database::get_companion_data() {
            Ok(c_d) => c_d,
            Err(e) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while getting companion data from sqlite database: {:?}", e)));
            },
        };
        Ok(companion_data)
    }

    #[staticmethod]
    fn fetch_user_data() -> PyResult<UserData> {
        let user_data: UserData =
        match Database::get_user_data() {
            Ok(u_d) => u_d,
            Err(e) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while getting user data from sqlite database: {:?}", e)));
            },
        };
        Ok(user_data)
    }

    #[staticmethod]
    fn change_first_message(new_first_message: String) -> PyResult<()> {
        match Database::change_first_message(&new_first_message) {
            Ok(_) => {},
            Err(e) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while changing companion's first message in sqlite database: {:?}", e)));
            },
        };
        Ok(())
    }

    #[staticmethod]
    fn change_companion_name(new_companion_name: String) -> PyResult<()> {
        match Database::change_companion_name(&new_companion_name) {
            Ok(_) => {},
            Err(e) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while changing companion name in sqlite database: {:?}", e)));
            },
        };
        Ok(())
    }

    #[staticmethod]
    fn change_user_name(new_user_name: String) -> PyResult<()> {
        match Database::change_username(&new_user_name) {
            Ok(_) => {},
            Err(e) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while changing username in sqlite database: {:?}", e)));
            },
        };
        Ok(())
    }

    #[staticmethod]
    fn change_companion_persona(new_companion_persona: String) -> PyResult<()> {
        match Database::change_companion_persona(&new_companion_persona) {
            Ok(_) => {},
            Err(e) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while changing companion persona in sqlite database: {:?}", e)));
            },
        };
        Ok(())
    }

    #[staticmethod]
    fn change_companion_example_dialogue(new_example_dialogue: String) -> PyResult<()> {
        match Database::change_companion_example_dialogue(&new_example_dialogue) {
            Ok(_) => {},
            Err(e) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while changing companion example dialogue in sqlite database: {:?}", e)));
            },
        };
        Ok(())
    }

    #[staticmethod]
    fn change_user_persona(new_user_persona: String) -> PyResult<()> {
        match Database::change_user_persona(&new_user_persona) {
            Ok(_) => {},
            Err(e) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while changing user persona in sqlite database: {:?}", e)));
            },
        };
        Ok(())
    }

    #[staticmethod]
    fn change_companion_data(new_companion_name: String, new_companion_persona: String, new_example_dialogue: String, new_first_message: String, long_term_memory_limit: u32, short_term_memory_limit: u32, roleplay: bool) -> PyResult<()> {
        match Database::change_companion(&new_companion_name, &new_companion_persona, &new_example_dialogue, &new_first_message, long_term_memory_limit, short_term_memory_limit, roleplay) {
            Ok(_) => {},
            Err(e) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while changing companion data in sqlite database: {:?}", e)));
            },
        }
        Ok(())
    }

    #[staticmethod]
    fn change_user_data(new_user_name: String, new_user_persona: String) -> PyResult<()> {
        match Database::change_user(&new_user_name, &new_user_persona) {
            Ok(_) => {},
            Err(e) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while changing user data in sqlite database: {:?}", e)));
            },
        };
        Ok(())
    }

    #[staticmethod]
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

    #[staticmethod]
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

    #[staticmethod]
    fn change_longterm_memory_limit(new_limit: u32) -> PyResult<()> {
        match Database::change_long_term_memory(new_limit) {
            Ok(_) => {},
            Err(e) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while changing long-term memory limit in sqlite database: {:?}", e)));
            }
        };
        Ok(())
    }

    #[staticmethod]
    fn change_shortterm_memory_limit(new_limit: u32) -> PyResult<()> {
        match Database::change_short_term_memory(new_limit) {
            Ok(_) => {},
            Err(e) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while changing short-term memory limit in sqlite database: {:?}", e)));
            },
        };
        Ok(())
    }

    #[staticmethod]
    fn change_roleplay(enable: bool) -> PyResult<()> {
        match Database::disable_enable_roleplay(enable) {
            Ok(_) => {},
            Err(e) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while enabling/disabling roleplay in sqlite database: {:?}", e)));
            },
        };
        Ok(())
    }

    #[staticmethod]
    fn import_character_json(character_json_text: String) -> PyResult<()> {
        let character_json: CharacterJson = match serde_json::from_str(&character_json_text) {
            Ok(v) => v,
            Err(e) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while parsing provided text as json: {:?}", e)));
            }
        };
        match Database::import_companion(&character_json.name, &character_json.description, &character_json.mes_example, &character_json.first_mes) {
            Ok(_) => Ok(()),
            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!("Error while importing character via character class to sqlite database {:?}", e))),
        }
    }

    #[staticmethod]
    fn import_character_card(character_card_path: &str) -> PyResult<()> {
        let decoder = png::Decoder::new(File::open(character_card_path)?);
        let reader = decoder.read_info().unwrap();
        let character_base64_option: Option<String> = reader.info().uncompressed_latin1_text.iter()
            .filter(|text_chunk| text_chunk.keyword == "chara")
            .map(|text_chunk| text_chunk.text.clone())
            .next();
            let character_base64: String = match character_base64_option {
                Some(v) => v,
                None => {
                    let mut f_buffer = Vec::new();
                    File::open(character_card_path)?.read_to_end(&mut f_buffer)?;
                    let text_chunk_start = f_buffer.windows(9).position(|window| window == b"tEXtchara").ok_or_else(|| pyo3::exceptions::PyValueError::new_err("No tEXt chunk with name 'chara' found"))?;
                    let text_chunk_end = f_buffer.windows(4).rposition(|window| window == b"IEND").ok_or_else(|| pyo3::exceptions::PyValueError::new_err("No tEXt chunk with name 'chara' found"))?;
                    String::from_utf8_lossy(&f_buffer[text_chunk_start + 10..text_chunk_end - 8]).to_string()
                }
            };
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

    #[staticmethod]
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

    #[staticmethod]
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

    #[staticmethod]
    fn get_character_json() -> PyResult<String> {
        let companion_data = match Database::get_companion_data() {
            Ok(m) => m,
            Err(e) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!("Error while fetching companion data as json: {:?}", e)));
            }
        };
        let character_data: CharacterJson = CharacterJson {
            name: companion_data.name,
            description: companion_data.persona,
            first_mes: companion_data.first_message,
            mes_example: companion_data.example_dialogue,
        };
        match serde_json::to_string_pretty(&character_data) {
            Ok(v) => Ok(v),
            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!("Error while encoding companion data as json: {:?}", e))),
        }
    }
}


#[pyfunction]
fn init() -> PyResult<Companion> {
    match Database::create() {
        Ok(_) => {},
        
        Err(e) => {
            let error_msg = format!("Error while connecting to sqlite database:, {}", e);
            return Err(pyo3::exceptions::PyValueError::new_err(error_msg)); }
    }

    match VectorDatabase::connect() {
        Ok(_) => { }
        Err(e) => { 
            let error_msg = format!("Error while connecting to long-term memory (tantivy): {}", e);
            return Err(pyo3::exceptions::PyValueError::new_err(error_msg)); }
    }

    Ok(Companion {
        ai_model: None,
        is_llama2: false,
    })
}

fn load_progress_callback(_: LoadProgress) {}

// works with https://zoltanai.github.io/character-editor/
// and with https://github.com/Hukasx0/aichar
#[derive(Serialize, Deserialize)]
struct CharacterJson {
    name: String,
    description: String,
    first_mes: String,
    mes_example: String,
}

#[derive(Deserialize)]
struct CharacterCard {
    name: String,
    description: String,
    first_mes: String,
    mes_example: String,
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

#[pymodule]
fn ai_companion_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(init, m)?)?;
    Ok(())
}
