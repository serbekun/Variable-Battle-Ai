ROUND_COUNT_LIMITS = 100
PLAYER_DEAD_HP = 0
BOT_DEAD_HP = 0

def show_display(player_hp, player_attack, player_heal, player_block, bot_hp, bot_attack, bot_heal, bot_block, round_count):
    print("turn -", round_count)
    print(f"Player hp - {player_hp} | Bot hp = {bot_hp}")
    print(f"Player attack = {player_attack} | Bot attack = {bot_attack}")
    print(f"Player heal = {player_heal} | Bot heal = {bot_heal}")
    print("block status", player_block)
    print("1. Attack\n2. Heal\n3. Block\n4. Increase attack\n5. Increase heal")

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
        print("you win nice")
        return False
    
    else:
        print("you can continue")
        return True