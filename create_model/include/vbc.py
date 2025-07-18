ROUND_COUNT_LIMITS = 100
PLAYER_DEAD_HP = 0
BOT_DEAD_HP = 0

def cat_limits(player_hp, player_attack, player_heal, bot_hp, bot_attack, bot_heal):

    if player_hp > 150:
        player_hp = 150
    if bot_hp > 150:
        bot_hp = 150

    if player_attack > 50:
        player_attack = 50
    if bot_attack > 50:
        bot_attack = 50

    if player_heal > 50:
        player_heal = 50
    if bot_heal > 50:
        bot_heal = 50

    return player_hp, player_attack, player_heal, bot_hp, bot_attack, bot_heal

def check_end_round(round_count, player_hp, bot_hp):

    if round_count > ROUND_COUNT_LIMITS:
        print("you lose because time limit")
        return False

    elif player_hp < PLAYER_DEAD_HP:
        print("you lose because you died")
        return False
    
    elif bot_hp < BOT_DEAD_HP:
        return False
    
    else:
        return True