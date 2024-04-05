use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};

use tch;
use tch::Tensor;

const TRAIN_FILE: &str = "Data/openwebtext_only_english/train.txt" ;
const VAL_FILE: &str = "Data/openwebtext_only_english/val.txt" ;
const BATCH_SIZE:i64 = 32;
const BLOCK_SIZE:i64 = 128;

fn create_stoi() -> HashMap<String, i64> {
    // Open the JSON file
    let mut file =  match File::open("Data/openwebtext_token/min_word_count.json"){
        Ok(file) => file,
        Err(error) => panic!("Problem opening the file: {:?}", error),
    };

    // Read the contents of the file into a string
    let mut contents = String::new();
    file.read_to_string(&mut contents).expect("TODO: panic message");

    // Parse the JSON string into a HashMap<String, i32>
    let stoi: HashMap<String, i64> = match serde_json::from_str(&contents){
        Ok(file) => file,
        Err(error) => panic!("Problem opening the file: {:?}", error),
    };
    return stoi;
}
//const STOI :HashMap<String, i32> = create_stoi();

// fn create_itos() -> HashMap<i64, String> {
//     let stoi = create_stoi();
//     let mut itos: HashMap<i64, String> = Default::default();
//     for (key, value) in stoi.iter() {
//
//         itos.insert(*value, String::from(key));
//     }
//     return itos;
//
// }
//const ITOS :HashMap<i32, String> = create_itos();


fn read_file(file_root: &str, file_pointer: u64) -> (Vec<i64>, u64) {

    let stoi = create_stoi();
    let mut temp_list: Vec<i64> = Vec::new();

    // Open the file
    let file =  File::open(file_root).expect("Cannot open file.txt");
    let mut reader = BufReader::new(file);

    // Seek to the specified position
    reader.seek(SeekFrom::Start(file_pointer)).expect("Cannot seek file pointer");
    for lines in reader.by_ref().lines() {
        for word in lines.unwrap().split_whitespace() {
            for keys in stoi.keys(){
                if word == keys{
                    let temp = stoi.get(word).unwrap();
                    temp_list.push(*temp);

                }
            }
        }
        if temp_list.len() >= (BLOCK_SIZE * BATCH_SIZE + 1) as usize {
            break
        }
    }let end_position = reader.seek(SeekFrom::Current(0)).expect("Cannot seek file pointer");
    return (temp_list, end_position);
}

fn encoded_data_batch(raw_data: Vec<i64>) -> Tensor {
    let mut tensor_list: Vec<Tensor> = Vec::new();
    let mut cat_list: Vec<Tensor> = Vec::new();
    let mut stack_list : Vec<Tensor> = Vec::new();

    for value in raw_data{
        let tensor = Tensor::from_slice(&[value]);
        tensor_list.push(tensor);
    }
    let mut counter = 0;
    for chunk in tensor_list{
        cat_list.push(chunk);
        counter +=1;
        if counter == BLOCK_SIZE+1{
            let concatenated_tensor = Tensor::cat(&cat_list, 0);
            //print!("{}", concatenated_tensor);
            stack_list.push(concatenated_tensor);
            cat_list.clear();
            counter = 0;
        }
    }
    let tensor_data = Tensor::stack(&stack_list, 0);
    //println!("{}", tensor_data.get(31));
    tensor_data

}


pub fn get_batch_train(new_file_pointer: u64) -> (Tensor, Tensor, u64) {
    let (train_data, new_file_pointer) = read_file(TRAIN_FILE, new_file_pointer);
    let  block_data = encoded_data_batch(train_data);
    let slice1 = block_data.narrow(1, 0, block_data.size()[1] - 1); // Equivalent to block_data[:, :-1] in python
    let slice2 = block_data.narrow(1, 1, block_data.size()[1] - 1); // Equivalent to block_data[:, 1:] in python
    return (slice1, slice2, new_file_pointer)


}
pub fn get_batch_val(new_file_pointer: u64) -> (Tensor, Tensor, u64) {
    let (train_data, _new_file_pointer) = read_file(VAL_FILE, new_file_pointer);
    let  block_data = encoded_data_batch(train_data);

    let slice1 = block_data.narrow(1, 0, block_data.size()[1] - 1); // Equivalent to block_data[:, :-1] in python
    let slice2 = block_data.narrow(1, 1, block_data.size()[1] - 1); // Equivalent to block_data[:, 1:] in python
    return (slice1, slice2, new_file_pointer)

}