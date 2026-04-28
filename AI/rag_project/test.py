import streamlit_authenticator as stauth
# Create the hasher object with the list of passwords
# Hasher is a utility class with class methods.
# We call hash_list directly without initializing the class.
passwords_to_hash = ['admin123']
hashed_passwords = stauth.Hasher.hash_list(passwords_to_hash)

print(hashed_passwords)