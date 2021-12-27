from main import User

def test_new_user():
    """
    GIVEN a User model
    WHEN a new User is created
    THEN check the email, password, firstname, Symptoms, Cancer, Treatment and results fields are defined correctly
    """
    user = User(email = 'jack123@gmail.com', password = '12345678ba', first_name='Jack',vCancer='YES',vTreatment='YES',vSymptoms='YES',result='50%')
    assert user.email == 'jack123@gmail.com'
    assert user.password == '12345678ba'
    assert user.first_name == 'Jack'
    assert user.vSymptoms == 'YES'
    assert user.vCancer == 'YES'
    assert user.vTreatment == 'YES'
    assert user.result == '50%'
    print("Test passed")


test_new_user()