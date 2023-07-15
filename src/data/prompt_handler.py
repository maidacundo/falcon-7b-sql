from typing import Optional
from data.utils import SQL_SPECIAL_TOKENS
from data.dataset_handler import SpiderSchema

def generate_prompt(examples, spider_schemas: Optional[SpiderSchema], use_fields=False):
    """
    Generates the prompt for the given example as follows:
    <|schema|> DATABASE STRUCTURE <|query|> NATURAL LANGUAGE QUESTION <|sql|> SQL QUERY <|endoftext|>

    examples: The example to generate the prompt for.
    spider_schemas: The spider schemas object.
    use_fields: Whether to use the database fields in the prompt.
    """
    prompt = ''

    if use_fields:
      fields = SQL_SPECIAL_TOKENS['schema'] + spider_schemas.get_db_schema(examples['db_id']) + SQL_SPECIAL_TOKENS['endoftext']
      fields.strip()
      prompt += fields

    question = SQL_SPECIAL_TOKENS['query'] + examples['question'] + SQL_SPECIAL_TOKENS['endoftext']
    question.strip()
    prompt += question

    query = SQL_SPECIAL_TOKENS['sql'] + examples['query']
    query.strip()

    if not query.endswith(';'):
        query = query + ';'
    query = query + SQL_SPECIAL_TOKENS['endoftext']

    prompt += query

    return prompt.lower()

def preprocess_function(examples, spider_schema, use_fields=False):
    prompts = generate_prompt(examples, spider_schema, use_fields)
    return {'input_text': prompts}
