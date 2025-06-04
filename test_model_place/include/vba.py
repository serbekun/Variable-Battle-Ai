import random

# Player increase
PLAYER_ATTACK_INC = 5
PLAYER_HEAL_INC = 5

# Bot increase
BOT_ATTACK_INC = 5
BOT_HEAL_INC = 5

# Players defs

def player_attack_def(player_attack, bot_hp, bot_block):

    if bot_block == True:
        bot_block = False
    else:
        bot_hp -= player_attack

    return bot_hp, bot_block
        
def player_heal_def(player_heal, player_hp):
    player_hp += player_heal
    return player_hp

def player_block_def(player_block):

    if player_block == True:
        return True
    
    block_chance = random.randint(1, 5)
    
    if block_chance != 5:
        player_block = True
    
    return player_block

def player_increase_attack_def(player_attack):
    player_attack += PLAYER_ATTACK_INC
    return player_attack

def player_increase_heal_def(player_heal):
    player_heal += PLAYER_HEAL_INC
    return player_heal

# Bot defs

def bot_attack_def(bot_attack, player_hp, player_block):

    if player_block == True:
        player_block = False
    else:
        player_hp -= bot_attack

    return player_hp, player_block

def bot_heal_def(bot_heal, bot_hp):
    bot_hp += bot_heal
    return bot_hp

def bot_block_def(bot_block):

    if bot_block == True:
        return True
    
    block_chance = random.randint(1, 5)
    
    if block_chance != 5:
        bot_block = True
    
    return bot_block

def bot_increase_attack_def(bot_attack):
    bot_attack += BOT_ATTACK_INC
    return bot_attack

def bot_increase_heal_def(bot_heal):
    bot_heal += BOT_HEAL_INC
    return bot_heal